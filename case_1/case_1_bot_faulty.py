import asyncio
import random
import traceback
import numpy as np
import pandas as pd
import collections
import os, sys

from datetime import datetime
from typing import Optional
from xchangelib import xchange_client, service_pb2 as utc_bot_pb2
from  prediction import Prediction
from grpc.aio import AioRpcError
def log_to_file(log_file_path):
    """
    Redirects the console output to a log file.
    Prints to both the console and the log file.
    """
    class Logger(object):
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log_file = open(log_file, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log_file.write(message)

        def flush(self):
            self.terminal.flush()
            self.log_file.flush()

    log_file_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)

    sys.stdout = Logger(log_file_path)


# constants
MAX_ORDER_SIZE = 40
MAX_OPEN_ORDERS = 10
OUTSTANDING_VOLUME = 120
MAX_ABSOLUTE_POSITION = 200
SYMBOLS = ["EPT", "DLO", "MKU", "IGM", "BRV"]
ETFS = ["SCP", "JAK"]
TRAP = 1234567890
df = pd.read_csv("Case1_Historical_Amended.csv")
class MyPositions:
    def __init__(self):
        self.positions = dict((symbol, 0) for symbol in SYMBOLS + ETFS)
        self.average_price = dict((symbol, 0) for symbol in SYMBOLS + ETFS)
    def update(self, symbol, price, qty):
        self.positions[symbol] += qty
        if self.positions[symbol] == 0:
            self.average_price[symbol] = 0
        else:
            self.average_price[symbol] = (self.average_price[symbol] * (self.positions[symbol] - qty) + price * qty) / self.positions[symbol]
    def get_position(self, symbol):
        return self.positions[symbol]
    def get_average_price(self, symbol):
        return self.average_price[symbol]

class OrderResponse:
    def __init__(self, order_id: str):
        self.order_id = order_id

class OpenOrders:
    def __init__(self):
        self.num_open_orders = 0
        self.outstanding_volume = 0
        self.trap_ids = set()
        self.id_to_price = {}
        self.id_to_qty = {}
        self.id_to_side = {}
        self.id_to_level = {}
        self.id_to_symbol = {}
        self.level_orders = dict((symbol, {0: 0, 1: 0, 2: 0, 3: 0}) for symbol in SYMBOLS + ETFS)
        self.queue = dict((symbol, collections.deque()) for symbol in SYMBOLS + ETFS)
    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj
        self.outstanding_volume += adj
        if self.id_to_qty[id] == 0:
            self.remove_order(id)
    def get_all_orders(self):
        return list(self.id_to_price.keys())
    def get_price(self, id):
        return self.id_to_price[id]
    def get_num_open_orders(self):
        return self.num_open_orders
    def get_qty(self, id):
        return self.id_to_qty[id]
    def get_symbol(self, id):
        return self.id_to_symbol[id]
    def get_side(self, id):
        if id not in self.id_to_side:
            return None
        return self.id_to_side[id]
    def get_level(self, id):
        return self.id_to_level[id]
    def get_symbol_levels(self, symbol):
        return self.level_orders[symbol]
    def get_outstanding_volume(self):
        return self.outstanding_volume
    def get_k_oldest_order(self, symbol, k):
        if len(self.queue[symbol]) >= k:
            count = 0
            res = []
            while count < k:
                id = self.queue[symbol].popleft()
                if id not in self.trap_ids:
                    res.append(id)
                    count += 1
            return res
        return [i for i in self.queue[symbol] if i not in self.trap_ids]
    def add_order(self, symbol, price, qty, id, side, level):
        self.id_to_price[id] = price
        self.id_to_qty[id] = qty
        self.id_to_side[id] = side
        self.id_to_level[id] = level
        self.id_to_symbol[id] = symbol
        self.num_open_orders += 1
        self.outstanding_volume += qty
        self.level_orders[symbol][level] += 1
        self.queue[symbol].append(id)

    def remove_order(self, id):
        if id not in self.id_to_level: return
        level = self.id_to_level[id]
        symbol = self.id_to_symbol[id]
        self.outstanding_volume -= self.id_to_qty[id]
        self.num_open_orders -= 1
        self.level_orders[symbol][level] -= 1
        if id in self.queue[symbol]:
            self.queue[symbol].remove(id)
        del self.id_to_price[id]
        del self.id_to_qty[id]
        del self.id_to_side[id]
        del self.id_to_level[id]
        

class MainBot(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str, open_orders, positions):
        super().__init__(host, username, password)
        self.order_size = 10
        self.level_orders = 10
        self.spreads = [2,4,6]
        self.open_orders_object = open_orders
        self.my_positions = positions
        self.open_orders = self.load_my_positions()
        self.predictors = [Prediction(symbol, df[symbol].to_numpy()) for symbol in SYMBOLS + ETFS]
        self.predictions = dict((pred.name(), 0) for pred in self.predictors)
        print("Object equality", self.open_orders_object)


    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        if success:
            self.writing_to_file(order_id, "CANCELLED")
            self.open_orders_object.remove_order(order_id)
        else:
            print(f"[DEBUG] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")
            self.open_orders_object.remove_order(order_id)

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        self.writing_to_file(order_id, "FILLED", price)
        self.my_positions.update(self.open_orders_object.get_symbol(order_id), price, qty if self.open_orders_object.get_side(order_id) == xchange_client.Side.BUY else -qty)
        self.open_orders_object.adjust_qty(order_id, -qty)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        self.writing_to_file(order_id, "REJECTED")
        self.open_orders_object.remove_order(order_id)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # print("something was traded")
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        # print("book update")
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # print("Swap response")
        pass

    async def bot_place_order(self, symbol, qty, side, price, level=0):
        if price == TRAP:
            order_id = await self.place_order(symbol, qty, side, price)
            self.open_orders_object.trap_ids.add(order_id)
            self.open_orders_object.add_order(symbol, price, qty, order_id, side, level)
            # self.writing_to_file(order_id, "PLACED")
            return order_id

        if level == 0:
            diff = self.open_orders_object.get_num_open_orders() + 1 - MAX_OPEN_ORDERS
            oldest_orders = self.open_orders_object.get_k_oldest_order(symbol, diff)
            for order_id in oldest_orders:
                # self.open_orders_object.remove_order(order_id)
                await self.cancel_order(order_id)
                
        vol = min(qty,
                  MAX_ORDER_SIZE,
                  MAX_ABSOLUTE_POSITION - self.positions[symbol] if side == xchange_client.Side.BUY else self.positions[symbol] + MAX_ABSOLUTE_POSITION,
                  OUTSTANDING_VOLUME - self.open_orders_object.get_outstanding_volume())
        if vol > 0:
            order_id = await self.place_order(symbol, vol, side, price)
            self.open_orders_object.add_order(symbol, price, vol, order_id, side, level)
            self.writing_to_file(order_id, "PLACED")
            return order_id
        
        
    async def bot_place_swap_order(self):
        pass

    async def bot_place_arbitrage_order(self, symbol):
        pass

    async def bot_handle_balancing_order(self, symbol):
        pass
    
    def writing_to_file(self, order_id, type, price = 0):
        verbose = True
        if not verbose: return
        if type == "FILLED":
            symbol = self.open_orders_object.get_symbol(order_id)
            side = self.open_orders_object.get_side(order_id)
            qty = self.open_orders_object.get_qty(order_id)
            gap = self.open_orders_object.get_price(order_id) - price
            with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
                f.write(f"{order_id} {(symbol, side)} {qty} {price} {gap}\n")
        elif type == "CANCELLED":
            symbol = self.open_orders_object.id_to_symbol[order_id]
            side = self.open_orders_object.get_side(order_id)
            with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
                f.write(f"[CANCELLED] {order_id} {(symbol, side)}\n")
        elif type == "REJECTED":
            symbol = self.open_orders_object.get_symbol(order_id)
            side = self.open_orders_object.get_side(order_id)
            with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
                f.write(f"[REJECTED] {order_id} {(symbol, side)}\n")
        elif type == "PLACED":
            symbol = self.open_orders_object.get_symbol(order_id)
            price = self.open_orders_object.get_price(order_id)
            level = self.open_orders_object.get_level(order_id)
            side = self.open_orders_object.get_side(order_id)
            with open(f"./log/placed/round_data_{start_time}.txt", "a") as f:
                f.write(f"{order_id} {level} {symbol} {side} {price}\n")

    def load_my_positions(self):
        open_orders = dict()
        for order_id in self.open_orders_object.get_all_orders():
            symbol = self.open_orders_object.get_symbol(order_id)
            qty = self.open_orders_object.get_qty(order_id)
            side = self.open_orders_object.get_side(order_id)
            side = utc_bot_pb2.NewOrderRequest.Side.BUY if side == xchange_client.Side.BUY else utc_bot_pb2.NewOrderRequest.Side.SELL
            price = self.open_orders_object.get_price(order_id)
            limit_order_msg = utc_bot_pb2.LimitOrder(qty=qty, px=price)
            order_request = utc_bot_pb2.NewOrderRequest(symbol=symbol, id=order_id, limit=limit_order_msg,
                                                        side=side)
            open_orders[order_id] = [order_request, qty, False]
        return open_orders
        # print(self.open_orders)
    async def bot_update_predictions(self):
        self.predictions = dict((pred.name(), pred.predict(2)) for pred in self.predictors)


    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        # intended to load the position if we are disconnected somehow
        self.load_my_positions()
        # Place a trap here LOL
        for symbol in SYMBOLS + ETFS:
            self.bot_place_order(symbol, 1, xchange_client.Side.SELL, TRAP)
        while True:
            for pred in self.predictors:
                order_book = self.order_books[pred.name()] if pred.name() in self.order_books else xchange_client.OrderBook()
                pred.update(order_book)
            await self.bot_update_predictions()
            bids = dict((pred.name(), pred.bid(self.predictions[pred.name()])) for pred in self.predictors)
            asks = dict((pred.name(), pred.ask(self.predictions[pred.name()])) for pred in self.predictors)
            # ETF Arbitrage
            # how aggressively to arbitrage
            rate = 0.8            
            for etf in ETFS:
                if etf == "SCP":
                    price = (3 * self.predictions["EPT"] + 3*self.predictions["IGM"] + 4*self.predictions["BRV"])/10
                elif etf == "JAK":
                    price = (2 * self.predictions['EPT'] + 5*self.predictions['DLO'] + 3*self.predictions['MKU'])/10
                margin = 40
                etf_bids = sorted([(k,v) for k, v in self.order_books[etf].bids.items() if k > price + margin and v > 0])
                etf_asks = sorted([(k,v) for k, v in self.order_books[etf].asks.items() if k < price - margin and v > 0], reverse=True)
                m = max(len(etf_bids), len(etf_asks))
                for i in range(m):
                    if i < len(etf_bids):
                        await self.bot_place_order(etf, int(rate * etf_bids[i][1]), xchange_client.Side.SELL, etf_bids[i][0])
                    if i < len(etf_asks):
                        await self.bot_place_order(etf, int(rate * etf_asks[i][1]), xchange_client.Side.BUY, etf_asks[i][0])

            # Take advantage of the spread
            # TODO: review
            for symbol in SYMBOLS:
                margin = 50
                symbol_bids = sorted([(k,v) for k, v in self.order_books[symbol].bids.items() if k > self.predictions[symbol] + margin and v > 0])
                symbol_asks = sorted([(k,v) for k, v in self.order_books[symbol].asks.items() if k < self.predictions[symbol] - margin and v > 0], reverse=True)
                m = max(len(symbol_bids), len(symbol_asks))
                for i in range(m):
                    if i < len(symbol_bids):
                        await self.bot_place_order(symbol, symbol_bids[i][1], xchange_client.Side.SELL, symbol_bids[i][0])
                    if i < len(symbol_asks):
                        await self.bot_place_order(symbol, symbol_asks[i][1], xchange_client.Side.BUY, symbol_asks[i][0])
            
            # Normal Trading
            for symbol, _ in self.predictions.items():
                if symbol in SYMBOLS + ETFS:
                    buy_volume, sell_volume = 5, 5
                    if self.positions[symbol] > MAX_ABSOLUTE_POSITION * 3/4:
                        price = self.my_positions.get_average_price(symbol)
                        if not price: price = self.predictions[symbol]
                        extreme_asks = sorted([(k,v) for k, v in self.order_books[symbol].asks.items() if k > price and v > 0])
                        for item in extreme_asks:
                            await self.bot_place_order(symbol, item[1], xchange_client.Side.SELL, item[0])
                    elif self.positions[symbol] < - MAX_ABSOLUTE_POSITION * 3/4:
                        price = abs(self.my_positions.get_average_price(symbol))
                        if not price: price = self.predictions[symbol]
                        extreme_bid = sorted([(k,v) for k, v in self.order_books[symbol].bids.items() if k < price and v > 0], reverse=True)
                        for item in extreme_bids:
                            await self.bot_place_order(symbol, item[1], xchange_client.Side.BUY, item[0])


                    if self.positions[symbol] > MAX_ABSOLUTE_POSITION //2:
                        # encourage selling
                        sell_volume = self.positions[symbol]//10
                        buy_volume = (MAX_ABSOLUTE_POSITION - self.positions[symbol])//10
                    elif self.positions[symbol] < - MAX_ABSOLUTE_POSITION//2:
                        # encourage buying
                        buy_volume = abs(self.positions[symbol])//10
                        sell_volume = (MAX_ABSOLUTE_POSITION + self.positions[symbol])//10
                    buy_first = random.choice([True, False])
                    if buy_first:
                        if int(bids[symbol]) > 0:
                            await self.bot_place_order(symbol, buy_volume + 1, xchange_client.Side.BUY, int(bids[symbol]))
                        elif int(asks[symbol]) > 0:
                            await self.bot_place_order(symbol, sell_volume + 1, xchange_client.Side.SELL, int(asks[symbol]))
                    else:
                        if int(asks[symbol]) > 0:
                            await self.bot_place_order(symbol, sell_volume + 1, xchange_client.Side.SELL, int(asks[symbol]))
                        elif int(bids[symbol]) > 0:
                            await self.bot_place_order(symbol, buy_volume + 1, xchange_client.Side.BUY, int(bids[symbol]))
  
            # Level Orders
            for symbol in SYMBOLS:
                for level in range(1, 4):
                    if bids[symbol] < 0 or asks[symbol] < 0:
                        continue
                    spread = self.spreads[level - 1]
                    bid = bids[symbol] - spread
                    ask = asks[symbol] + spread

                    if self.open_orders_object.get_symbol_levels(symbol)[level] < self.level_orders:
                        await self.bot_place_order(symbol, 2, xchange_client.Side.BUY, int(bid), level)
                    if self.open_orders_object.get_symbol_levels(symbol)[level] < self.level_orders:
                        await self.bot_place_order(symbol, 2, xchange_client.Side.SELL, int(ask), level)
            # Viewing Positions
            print("My positions:", self.positions)
            await asyncio.sleep(1)

    async def view_books(self):
        """Prints the books every 3 seconds."""
        while True:
            await asyncio.sleep(3)
            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self):
        """
        Creates tasks that can be run in the background. Then connects to the exchange
        and listens for messages.
        """
        asyncio.create_task(self.trade())
        await self.connect()
        # asyncio.create_task(self.view_books())


async def main():
    count = 0
    open_orders = OpenOrders()
    my_positions = MyPositions()
    while True:
        # log_file_path = f"/Users/divy/Desktop/Chicago-Trading-Competition-2024/case_1/log/file{count}.txt"
        # log_to_file(log_file_path)
        bot = MainBot("staging.uchicagotradingcompetition.com:3333", "university_of_chicago_umassamherst", "ekans-mew-8133", open_orders=open_orders, positions=my_positions)
        count += 1
        try:
            await bot.start()
            await asyncio.Event().wait()
        except AioRpcError as e:
            print(f"ConnectionError occurred: {e.with_traceback(None)}")
            open_orders = OpenOrders()
            my_positions = MyPositions()
            await asyncio.sleep(1)
        except Exception as e:
            traceback.print_exc()
            print(f"Exception occurred: {e.with_traceback(None)}")  # Print the traceback
            print("Restarting the bot...")
            await asyncio.sleep(1)  # Wait for a short duration before restarting
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Closing the event loop...")
            break

if __name__ == "__main__":
    start_time = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    asyncio.run(main())
    


