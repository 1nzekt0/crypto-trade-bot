import time
import json
import logging
import datetime
import pandas as pd
import numpy as np
import traceback
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance.enums import *

# Configuración del sistema de logging
def setup_logger():
    logger = logging.getLogger('crypto_trade_bot')
    logger.setLevel(logging.INFO)
    
    # Handler para la consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Handler para archivo
    file_handler = logging.FileHandler('crypto_bot.log')
    file_handler.setLevel(logging.INFO)
    
    # Formato del log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Agregar handlers al logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Cargar configuración desde el archivo config.json
def load_config():
    try:
        with open('config.json', 'r') as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        logger.error("Archivo config.json no encontrado")
        raise
    except json.JSONDecodeError:
        logger.error("Error al decodificar el archivo config.json")
        raise

# Clase principal del bot de trading
class CryptoTradeBot:
    def __init__(self, api_key, api_secret, config):
        self.config = config
        self.logger = logging.getLogger('crypto_trade_bot')
        
        # Inicializar cliente de Binance
        try:
            self.client = Client(api_key, api_secret)
            self.logger.info("Cliente de Binance inicializado correctamente")
        except Exception as e:
            self.logger.error(f"Error al inicializar el cliente de Binance: {str(e)}")
            raise
        
        # Parámetros del bot
        self.symbol = config['trading_pair']
        self.timeframe = config['timeframe']
        self.rsi_period = config['rsi_period']
        self.rsi_overbought = config['rsi_overbought']
        self.rsi_oversold = config['rsi_oversold']
        self.bollinger_period = config['bollinger_period']
        self.bollinger_std = config['bollinger_std']
        self.initial_balance = config['initial_balance']
        self.position = {
            'in_position': False,
            'quantity': 0,
            'buy_price': 0
        }
        
        # Registro de transacciones
        self.transactions = []

    # Obtener datos históricos de Binance
    def get_historical_data(self):
        try:
            # Convertir el timeframe a formato Binance
            timeframe_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            
            binance_timeframe = timeframe_map.get(self.timeframe, Client.KLINE_INTERVAL_1HOUR)
            
            # Obtener datos históricos
            klines = self.client.get_historical_klines(
                self.symbol, 
                binance_timeframe, 
                f"{self.bollinger_period * 3} {self.timeframe} ago UTC"
            )
            
            # Convertir a DataFrame
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos de datos
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data['close'] = data['close'].astype(float)
            
            self.logger.info(f"Datos históricos obtenidos correctamente. Filas: {len(data)}")
            return data
        
        except BinanceAPIException as e:
            self.logger.error(f"Error de API Binance: {str(e)}")
            return None
        except BinanceRequestException as e:
            self.logger.error(f"Error de solicitud a Binance: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error desconocido al obtener datos históricos: {str(e)}")
            return None

    # Calcular RSI
    def calculate_rsi(self, data, period=14):
        delta = data['close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    # Calcular Bandas de Bollinger
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        data['sma'] = data['close'].rolling(window=period).mean()
        data['std'] = data['close'].rolling(window=period).std()
        data['upper_band'] = data['sma'] + (data['std'] * std_dev)
        data['lower_band'] = data['sma'] - (data['std'] * std_dev)
        
        return data

    # Ejecutar orden de compra
    def execute_buy_order(self, price):
        try:
            balance = float(self.client.get_asset_balance(asset='USDT')['free'])
            
            if balance < self.config['min_order_value']:
                self.logger.warning(f"Balance insuficiente para comprar: {balance} USDT")
                return False
            
            # Calcular cantidad a comprar
            order_value = min(balance, self.config['max_order_value'])
            quantity = order_value / price
            
            # Redondear la cantidad según las reglas del mercado
            info = self.client.get_symbol_info(self.symbol)
            lot_size_filter = next(filter(lambda f: f['filterType'] == 'LOT_SIZE', info['filters']))
            step_size = float(lot_size_filter['stepSize'])
            precision = int(round(-math.log10(step_size)))
            quantity = round(quantity, precision)
            
            if quantity * price < self.config['min_order_value']:
                self.logger.warning(f"Orden demasiado pequeña: {quantity} {self.symbol} @ {price}")
                return False
            
            # Ejecutar la orden
            order = self.client.create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            # Actualizar posición
            self.position['in_position'] = True
            self.position['quantity'] = quantity
            self.position['buy_price'] = price
            
            # Registrar transacción
            transaction = {
                'type': 'BUY',
                'timestamp': datetime.datetime.now().isoformat(),
                'symbol': self.symbol,
                'price': price,
                'quantity': quantity,
                'value': quantity * price
            }
            self.transactions.append(transaction)
            
            self.logger.info(f"Orden de compra ejecutada: {quantity} {self.symbol} @ {price}")
            return True
            
        except BinanceAPIException as e:
            self.logger.error(f"Error de API Binance al comprar: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error desconocido al comprar: {str(e)}")
            traceback.print_exc()
            return False

    # Ejecutar orden de venta
    def execute_sell_order(self, price):
        try:
            if not self.position['in_position']:
                self.logger.warning("Intento de venta sin posición abierta")
                return False
            
            # Ejecutar la orden
            order = self.client.create_order(
                symbol=self.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=self.position['quantity']
            )
            
            # Calcular ganancia/pérdida
            buy_value = self.position['quantity'] * self.position['buy_price']
            sell_value = self.position['quantity'] * price
            profit = sell_value - buy_value
            profit_percent = (profit / buy_value) * 100
            
            # Registrar transacción
            transaction = {
                'type': 'SELL',
                'timestamp': datetime.datetime.now().isoformat(),
                'symbol': self.symbol,
                'price': price,
                'quantity': self.position['quantity'],
                'value': sell_value,
                'profit': profit,
                'profit_percent': profit_percent
            }
            self.transactions.append(transaction)
            
            # Actualizar posición
            self.position['in_position'] = False
            self.position['quantity'] = 0
            self.position['buy_price'] = 0
            
            self.logger.info(f"Orden de venta ejecutada: {transaction['quantity']} {self.symbol} @ {price}. Profit: {profit_percent:.2f}%")
            return True
            
        except BinanceAPIException as e:
            self.logger.error(f"Error de API Binance al vender: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error desconocido al vender: {str(e)}")
            traceback.print_exc()
            return False

    # Actualizar registro de transacciones
    def save_transactions(self):
        try:
            with open('transactions.json', 'w') as f:
                json.dump(self.transactions, f, indent=4)
            self.logger.info("Registro de transacciones guardado")
        except Exception as e:
            self.logger.error(f"Error al guardar transacciones: {str(e)}")

    # Análisis técnico y decisión de trading
    def analyze_and_trade(self):
        # Obtener datos históricos
        data = self.get_historical_data()
        if data is None or len(data) < self.bollinger_period:
            self.logger.warning("Datos insuficientes para análisis")
            return
        
        # Calcular indicadores
        data['rsi'] = self.calculate_rsi(data, self.rsi_period)
        data = self.calculate_bollinger_bands(data, self.bollinger_period, self.bollinger_std)
        
        # Obtener última fila de datos
        latest = data.iloc[-1]
        
        # Obtener precio actual
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
        except Exception as e:
            self.logger.error(f"Error al obtener precio actual: {str(e)}")
            return
        
        # Lógica de trading
        self.logger.info(f"Análisis para {self.symbol}: Precio={current_price}, RSI={latest['rsi']:.2f}, "
                         f"BB Superior={latest['upper_band']:.2f}, BB Inferior={latest['lower_band']:.2f}")
        
        # Señal de compra: RSI por debajo del nivel de sobreventa y precio cerca o por debajo de la banda inferior
        if not self.position['in_position'] and latest['rsi'] < self.rsi_oversold and current_price <= latest['lower_band'] * 1.02:
            self.logger.info(f"Señal de COMPRA detectada: RSI={latest['rsi']:.2f}, Precio={current_price}")
            self.execute_buy_order(current_price)
        
        # Señal de venta: RSI por encima del nivel de sobrecompra o precio cerca o por encima de la banda superior
        elif self.position['in_position'] and (latest['rsi'] > self.rsi_overbought or current_price >= latest['upper_band'] * 0.98):
            profit_percent = ((current_price - self.position['buy_price']) / self.position['buy_price']) * 100
            self.logger.info(f"Señal de VENTA detectada: RSI={latest['rsi']:.2f}, Precio={current_price}, Ganancia potencial={profit_percent:.2f}%")
            self.execute_sell_order(current_price)
        
        # También vender si hay una pérdida significativa para gestionar el riesgo
        elif self.position['in_position']:
            loss_percent = ((current_price - self.position['buy_price']) / self.position['buy_price']) * 100
            if loss_percent < -self.config['max_loss_percent']:
                self.logger.warning(f"Stop loss activado: Pérdida={loss_percent:.2f}%")
                self.execute_sell_order(current_price)
        
        # Guardar transacciones después de cada análisis
        self.save_transactions()

    # Bucle principal del bot
    def run(self):
        self.logger.info(f"Iniciando bot de trading para {self.symbol}")
        
        try:
            while True:
                self.analyze_and_trade()
                
                # Esperar según el intervalo configurado
                sleep_time = self.config.get('check_interval', 60)  # Por defecto, 60 segundos
                self.logger.info(f"Esperando {sleep_time} segundos hasta el próximo análisis...")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Bot detenido manualmente")
        except Exception as e:
            self.logger.error(f"Error inesperado: {str(e)}")
            traceback.print_exc()
        finally:
            self.logger.info("Bot finalizado")
            self.save_transactions()

# Punto de entrada principal
if __name__ == "__main__":
    # Configurar logger
    logger = setup_logger()
    logger.info("Iniciando CryptoTradeBot")
    
    try:
        # Cargar configuración
        config = load_config()
        
        # Inicializar bot
        bot = CryptoTradeBot(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            config=config
        )
        
        # Ejecutar bot
        bot.run()
        
    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        traceback.print_exc()
