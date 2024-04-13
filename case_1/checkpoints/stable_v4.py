import asyncio
import math
import random
import traceback
import numpy as np
import pandas as pd
import collections
import os, sys
import importlib
import tkinter as tk
from tkinter import ttk

import params
from params_gui import ParametersGUI
from datetime import datetime
from typing import Optional
from xchangelib import xchange_client, service_pb2 as utc_bot_pb2
from  prediction import Prediction
from grpc.aio import AioRpcError
import random

random.seed(1)
# constants
MAX_ORDER_SIZE = 40
MAX_OPEN_ORDERS = 50
OUTSTANDING_VOLUME = 120
MAX_ABSOLUTE_POSITION = 200
SYMBOLS = ["EPT", "DLO", "MKU", "IGM", "BRV"]
ETFS = ["SCP", "JAK"]
ASSET = ["JMS"]
TRAP = 8500
df = pd.read_csv("Case1_Historical_Amended.csv")

class OrderResponse:
    def __init__(self, order_id: str):
        self.order_id = order_id

class OpenOrders:
    def __init__(self):
        self.num_open_orders = dict((symbol, 0) for symbol in SYMBOLS + ETFS + ASSET)
        self.outstanding_volume = dict((symbol, 0) for symbol in SYMBOLS + ETFS + ASSET)
        self.id_to_price = {}
        self.id_to_qty = {}
        self.id_to_side = {}
        self.id_to_level = {}
        self.id_to_symbol = {}
        self.level_orders = dict((symbol, {0: 0, 1: 0, 2: 0, 3: 0}) for symbol in SYMBOLS + ETFS + ASSET)
        self.queue = dict((symbol, collections.deque()) for symbol in SYMBOLS + ETFS + ASSET)
        self.trap_ids = set()
    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj
        symbol = self.id_to_symbol[id]
        self.outstanding_volume[symbol] += adj
        if self.id_to_qty[id] == 0:
            self.remove_order(id)
    def get_all_orders(self):
        return list(self.id_to_price.keys())
    def get_price(self, id):
        return self.id_to_price[id]
    def get_num_open_orders(self, symbol):
        return self.num_open_orders[symbol]
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
    def get_outstanding_volume(self, symbol):
        return self.outstanding_volume[symbol]
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
        self.num_open_orders[symbol] += 1
        self.outstanding_volume[symbol] += qty
        self.level_orders[symbol][level] += 1
        self.queue[symbol].append(id)

    def remove_order(self, id):
        if id not in self.id_to_level: return
        level = self.id_to_level[id]
        symbol = self.id_to_symbol[id]
        self.outstanding_volume[symbol] -= self.id_to_qty[id]
        self.num_open_orders[symbol] -= 1
        self.level_orders[symbol][level] -= 1
        if id in self.queue[symbol]:
            self.queue[symbol].remove(id)
        del self.id_to_price[id]
        del self.id_to_qty[id]
        del self.id_to_side[id]
        del self.id_to_level[id]
        

class MainBot(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str, open_orders):
        super().__init__(host, username, password)
        self.round = 0
        self.safety_check = 0
        self.order_size = 16
        self.level_orders = 10
        self.fifty_threshold = {
            "EPT": 0.5,
            "DLO": 0.5,
            "MKU": 0.5,
            "IGM": 0.5,
            "BRV": 0.5,
            "SCP": 0.5,
            "JAK": 0.5,
        }

        self.etf_margin = NotImplemented
        self.spreads = NotImplemented
        self.fade = NotImplemented
        self.min_margin = NotImplemented
        self.edge_sensitivity = NotImplemented # keep between 0 (broad) and 1 (steep). roughly decides the steepness of the edge and therefore the range of num orders around it; edge sensitivity = k => 4/k orders in range
        self.slack = NotImplemented # (edge range = [min_margin, min_margin + slack])

        self.profit = 0
        self.open_orders_object = open_orders
        self.open_orders = self.load_open_orders()
        self.last_transacted_price = dict((symbol, {side: {
            "price": 0, "age": 0
        } for side in [xchange_client.Side.BUY, xchange_client.Side.SELL]}) for symbol in SYMBOLS + ETFS + ASSET)
        self.fade_augmented = dict((symbol, 0) for symbol in SYMBOLS + ETFS + ASSET)
        self.edge_augmented = dict((symbol, {side: 0 for side in [xchange_client.Side.BUY, xchange_client.Side.SELL]}) for symbol in SYMBOLS + ETFS + ASSET)
        self.history = dict((symbol, []) for symbol in SYMBOLS + ETFS + ASSET)

    def get_last_transacted_price(self, symbol, side):
        return self.last_transacted_price[symbol][side]["price"]
    
    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        if success:
            self.writing_to_file(order_id, "CANCELLED")
            self.open_orders_object.remove_order(order_id)
        else:
            pass
            #print(f"[DEBUG] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")
            # self.open_orders_object.remove_order(order_id)

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        self.writing_to_file(order_id, "FILLED", price)
        if self.open_orders_object.get_level(order_id) == 0:
            self.last_transacted_price[self.open_orders_object.get_symbol(order_id)][self.open_orders_object.get_side(order_id)]["price"] = price
            self.last_transacted_price[self.open_orders_object.get_symbol(order_id)][self.open_orders_object.get_side(order_id)]["age"] = self.round
        self.history[self.open_orders_object.get_symbol(order_id)].append(price)
        self.profit += self.calculate_profit(self.open_orders_object.get_side(order_id), qty, price)
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

    async def bot_place_order(self, symbol, qty, side, price=0, level=0, market=False):
        vol = min(qty,
                  MAX_ORDER_SIZE,
                  MAX_ABSOLUTE_POSITION - self.positions[symbol] if side == xchange_client.Side.BUY else self.positions[symbol] + MAX_ABSOLUTE_POSITION,
                  OUTSTANDING_VOLUME - self.open_orders_object.get_outstanding_volume(symbol))
        if vol <= 0:
            return 0
        if level == 0:
            diff = self.open_orders_object.get_num_open_orders(symbol) + 1 - MAX_OPEN_ORDERS
            oldest_orders = self.open_orders_object.get_k_oldest_order(symbol, diff)
            for order_id in oldest_orders:
                await self.cancel_order(order_id)
        if market==True:
            order_id = await self.place_order(symbol, qty, side)
            self.open_orders_object.add_order(symbol, price, qty, order_id, side, level)
            self.writing_to_file(order_id, "PLACED")
            return order_id
        if self.open_orders_object.get_num_open_orders(symbol) >= MAX_OPEN_ORDERS:
            return 
        
        if vol > 0:
            order_id = await self.place_order(symbol, vol, side, price)
            self.open_orders_object.add_order(symbol, price, vol, order_id, side, level)
            self.writing_to_file(order_id, "PLACED")
            return order_id
        
        
    async def bot_place_swap_order(self, swap, qty):
        return await self.place_swap_order(swap, qty)
        

    async def bot_place_arbitrage_order(self, etf, side, fair):
        qty:int
        convert = {
            "SCP":{"EPT":3, "IGM":3, "BRV":4},
            "JAK":{"EPT":2, "DLO":5, "MKU":3}
        }
        # BUY IN
        # qty = min([(MAX_ABSOLUTE_POSITION - self.positions[symbol])//convert[etf][symbol] for symbol in convert[etf]] +
        #               [(MAX_ABSOLUTE_POSITION - self.positions[etf])//10])
        qty = random.randint(1, 3)
        if side == "from":
            await self.bot_place_order(etf, qty * 10, xchange_client.Side.BUY, round(fair[etf]))
        elif side == "to":
            for symbol in convert[etf]:
                await self.bot_place_order(symbol, qty * 10, xchange_client.Side.BUY, round(fair[symbol]))
        
        # SWAP
        await self.bot_place_swap_order(f"{side}{etf}", qty)
        
        
        if side == "from":
            for symbol in convert[etf]:
                await self.bot_place_order(symbol, qty * 10, xchange_client.Side.SELL, round(fair[symbol]))
        elif side == "to":
            await self.bot_place_order(etf, qty *10, xchange_client.Side.SELL, round(fair[etf]))
        
        # After swapping, sell the newly converted values

        if side == "from":
            for symbol in convert[etf]:
                await self.bot_place_order(symbol, qty, xchange_client.Side.SELL, round(fair[symbol]))
        elif side == "to":
            await self.bot_place_order(etf, qty, xchange_client.Side.SELL, round(fair[etf]))
        return qty

    async def bot_handle_balancing_order(self, symbol):
        pass
    
    def writing_to_file(self, order_id, type, price = 0):
        verbose =True
        if not verbose: return
        if type == "FILLED":
            symbol = self.open_orders_object.get_symbol(order_id)
            side = self.open_orders_object.get_side(order_id)
            qty = self.open_orders_object.get_qty(order_id)
            gap = self.open_orders_object.get_price(order_id) - price
            with open(f"./log/filled/round_data_{start_time}.txt", "a") as f:
                f.write(f"{order_id} {(symbol, side)} {qty} {price} {gap} | Profit: {self.profit}\n")
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

    def load_open_orders(self):
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
    
    def set_fade_logic(self):
        for symbol in SYMBOLS + ETFS:
            absolute_position = abs(self.positions[symbol]) / MAX_ABSOLUTE_POSITION
            sign = 1 if self.positions[symbol] > 0 else -1
            self.fade_augmented[symbol] = - self.fade * sign *math.log2(1 + absolute_position)

    def get_market_activity_level(self, symbol, side):
        price = self.get_last_transacted_price(symbol, side)
        side_orders = self.order_books[symbol].bids if side == xchange_client.Side.BUY else self.order_books[symbol].asks
        # count the number of orders within the edge_window
        count = 0
        edge_window = self.min_margin + self.slack
        for order in side_orders:
            if abs(order - price) < edge_window:
                count += 1
        return count

    
    def set_edge_logic(self):
        for symbol in SYMBOLS + ETFS:
            for side in [xchange_client.Side.BUY, xchange_client.Side.SELL]:
                amplitude = self.slack/2
                min_margin = self.min_margin
                activity_level = self.get_market_activity_level(symbol, side)
                # 4 is magic number. At sensitivity = 1, it will be so steep that one order in window will set edge = 1
                self.edge_augmented[symbol][side] = max(int(round(min_margin + (amplitude * math.tanh(-4*self.edge_sensitivity * activity_level+2) + 1))) + random.randrange(-1, 1), 1)
                #print("edge: ", self.edge_augmented[symbol][side])


    def calculate_profit(self, side, qty, price):
        return -price * qty if side == xchange_client.Side.BUY else price * qty
    def update_safety_check(self):
        estimated_pnl = self.estimate_pnl()
        old_safety_check = self.safety_check
        if estimated_pnl >= 150000:
            self.safety_check += 1
        elif estimated_pnl >= 100000:
            self.safety_check += 0.8
        elif estimated_pnl >= 50000:
            self.safety_check += 0.5
        elif estimated_pnl >= 10000:
            self.safety_check += 0.2
        else:
            self.safety_check = 0
        return old_safety_check
    
    async def revert_to_balance(self, bids, asks):
        for symbol in SYMBOLS + ETFS:
            if self.positions[symbol] > 0:
                await self.bot_place_order(symbol, self.positions[symbol], xchange_client.Side.SELL, asks[symbol])
            if self.positions[symbol] < 0:
                await self.bot_place_order(symbol, -self.positions[symbol], xchange_client.Side.BUY, bids[symbol])
    
    def load_params(self):
        importlib.reload(params)
        params_instance = params.get_params()
        self.min_margin = params_instance.contract_params["min_margin"]
        self.fade = params_instance.contract_params["fade"]
        self.edge_sensitivity = params_instance.contract_params["edge_sensitivity"]
        self.slack = params_instance.contract_params["slack"]
        self.spreads = params_instance.spreads
        self.etf_margin = params_instance.etf_margin
        self.safety = params_instance.safety
        print(f"Parameters loaded: {self.min_margin}, {self.fade}, {self.edge_sensitivity}, {self.slack}")

    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        # intended to load the position if we are disconnected somehow
        self.load_params()
        self.load_open_orders()

        # buy as much JMS as possible
        await self.bot_place_order("JMS", OUTSTANDING_VOLUME, xchange_client.Side.BUY, 4999)

        #place bogus bids

        # get first round prices
        for symbol in SYMBOLS + ETFS:
            await self.bot_place_order(symbol, 1, xchange_client.Side.SELL, 0, market=True)
            await self.bot_place_order(symbol, 1, xchange_client.Side.BUY, 0, market=True)
            self.round += 1
            await asyncio.sleep(1)


        while True:
            for symbol in SYMBOLS + ETFS:
                await self.bot_place_order(symbol, 1, xchange_client.Side.BUY, 0)
                await self.bot_place_order(symbol, 1, xchange_client.Side.SELL, TRAP)

            self.load_params()
            self.set_fade_logic()
            self.set_edge_logic()

            #print(f"Slack is {self.slack}", f"Fade is {self.fade}", f"Edge Sensitivity is {self.edge_sensitivity}")
            
            bids = dict((symbol, self.get_last_transacted_price(symbol, xchange_client.Side.BUY) + self.fade_augmented[symbol] - self.edge_augmented[symbol][xchange_client.Side.BUY]) for symbol in SYMBOLS + ETFS)
            asks = dict((symbol, self.get_last_transacted_price(symbol, xchange_client.Side.SELL) + self.fade_augmented[symbol] + self.edge_augmented[symbol][xchange_client.Side.SELL]) for symbol in SYMBOLS + ETFS)
        
            # check for safety before trading
            if self.safety == True:
                self.update_safety_check()
                if self.safety_check >=5:
                    #end the round
                    self.revert_to_balance(bids, asks)
                    self.round += 1
                    print(self.estimate_pnl())
                    await asyncio.sleep(1)
                    continue                
            # handle the unbalanced position
            # ETF Arbitrage
            # TODO: review

            # If it has been too long since last order, sacrifice to get true price
            for symbol in SYMBOLS + ETFS:
                print("Age of ", symbol, self.last_transacted_price[symbol][xchange_client.Side.BUY]["age"])
                if self.round - self.last_transacted_price[symbol][xchange_client.Side.BUY]["age"] > 10:
                    await self.bot_place_order(symbol, 1, xchange_client.Side.BUY, 0, market=True)
                if self.round - self.last_transacted_price[symbol][xchange_client.Side.SELL]["age"] > 10:
                    await self.bot_place_order(symbol, 1, xchange_client.Side.SELL, 0, market=True)

            for etf in ETFS:
                fair = dict((symbol, np.mean(self.history[symbol][-5:]) if len(self.history[symbol]) >= 5 else np.mean(self.history[symbol])) for symbol in SYMBOLS + ETFS)
                if etf == "SCP":
                    price = (3 * fair["EPT"] + 3*fair["IGM"] + 4*fair["BRV"])/10
                if etf == "JAK":
                    price = (2 * fair['EPT'] + 5*fair['DLO'] + 3*fair['MKU'])/10
                predicted_price = fair[etf]
                if predicted_price - price > self.etf_margin:
                    with open(f"./log/etfs/round_data_{start_time}.txt", "a") as f:
                        f.write(f"ETF: {etf} | Predicted: {predicted_price} | Actual: {price} | Margin: {predicted_price - price}\n")
                    await self.bot_place_arbitrage_order(etf, "to", fair)
                elif predicted_price - price < - self.etf_margin:
                    with open(f"./log/etfs/round_data_{start_time}.txt", "a") as f:
                        f.write(f"ETF: {etf} | Predicted: {predicted_price} | Actual: {price} | Margin: {predicted_price - price}\n")
                    await self.bot_place_arbitrage_order(etf, "from", fair)
            
            # Penny In Penny Out
            for symbol in SYMBOLS + ETFS:
                buy_volume = random.randint(3, 9)
                sell_volume = buy_volume
                buy_first = random.choice([True, False])
                # print(bids[symbol], asks[symbol])
                bid = min(round(bids[symbol]), round(asks[symbol]))
                ask = max(round(bids[symbol]), round(asks[symbol]))
                if buy_first:
                    if int(bids[symbol]) > 0:
                        await self.bot_place_order(symbol, buy_volume, xchange_client.Side.BUY, bid)
                    if int(asks[symbol]) > 0:
                        await self.bot_place_order(symbol, sell_volume, xchange_client.Side.SELL, ask)
                else:
                    if int(asks[symbol]) > 0:
                        await self.bot_place_order(symbol, sell_volume, xchange_client.Side.SELL, ask)
                    if int(bids[symbol]) > 0:
                        await self.bot_place_order(symbol, buy_volume, xchange_client.Side.BUY, bid)
  
            # # Level Orders
            # TODO: review
            for symbol in SYMBOLS:
                for level in range(1, 3):
                    if bids[symbol] < 0 or asks[symbol] < 0:
                        continue
                    spread = self.spreads[level - 1]
                    bid = bids[symbol] - spread
                    ask = asks[symbol] + spread
                    buy_first = random.choice([True, False])
                    if buy_first:
                        if self.open_orders_object.get_symbol_levels(symbol)[level] < self.level_orders:
                            await self.bot_place_order(symbol, 3, xchange_client.Side.BUY, round(bid), level)
                        if self.open_orders_object.get_symbol_levels(symbol)[level] < self.level_orders:
                            await self.bot_place_order(symbol, 3, xchange_client.Side.SELL, round(ask), level)
                    else:
                        if self.open_orders_object.get_symbol_levels(symbol)[level] < self.level_orders:
                            await self.bot_place_order(symbol, 3, xchange_client.Side.SELL, round(ask), level)
                        if self.open_orders_object.get_symbol_levels(symbol)[level] < self.level_orders:
                            await self.bot_place_order(symbol, 3, xchange_client.Side.BUY, round(bid), level)
            # Viewing Positions
            print("Self: ", self.estimate_pnl())
            # print("My positions:", self.positions)
            self.round += 1
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
    while True:
        bot = MainBot("staging.uchicagotradingcompetition.com:3333", "university_of_chicago_umassamherst", "ekans-mew-8133", open_orders=open_orders)
        count += 1
        try:
            await bot.start()
            await asyncio.Event().wait()
        except AioRpcError as e:
            print(f"ConnectionError occurred: {e.with_traceback(None)}")
            open_orders = OpenOrders()
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
    

