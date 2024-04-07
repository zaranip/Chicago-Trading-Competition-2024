from datetime import datetime
from typing import Optional
from xchangelib import xchange_client
from  prediction import Prediction
import asyncio
import numpy as np
import pandas as pd

class OrderResponse:
    def __init__(self, order_id: str):
        self.order_id = order_id

class OpenOrders:
    def __init__(self, contract):
        self.contract_name = contract
        self.num_open_orders = 0
        self.price_to_id = {}
        self.id_to_price = {}
        self.id_to_qty = {}

    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj
        if self.id_to_qty[id] == 0:
            self.num_open_orders -= 1
            price = self.id_to_price[id]
            del self.id_to_price[id]
            del self.price_to_id[price]
            del self.id_to_qty[id]

    def modify_order(self, price, qty, old_id, new_id):
        if old_id == new_id:
            if old_id not in self.id_to_price:
                self.id_to_price[old_id] = price
                self.price_to_id[price] = old_id
                self.id_to_qty[old_id] = qty
                self.num_open_orders += 1
            else:
                del self.price_to_id[self.id_to_price[old_id]]
                self.price_to_id[price] = old_id
                self.id_to_price[old_id] = price
                self.id_to_qty[old_id] = qty
        else:
            if old_id not in self.id_to_price:
                self.id_to_price[new_id] = price
                self.price_to_id[price] = new_id
                self.id_to_qty[new_id] = qty
                self.num_open_orders += 1
            else:
                del self.price_to_id[self.id_to_price[old_id]]
                del self.id_to_price[old_id]
                del self.id_to_qty[old_id]
                self.price_to_id[price] = new_id
                self.id_to_price[new_id] = price
                self.id_to_qty[new_id] = qty

    def get_qty(self, price):
        p_id = self.price_to_id[price]
        return self.id_to_qty[p_id]

    def get_id(self, price):
        return self.price_to_id[price]

class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.contracts = ["EPT", "DLO", "MKU", "IGM", "BRV"]
        self.order_size = 10
        self.order_l1 = 15
        self.order_l2 = 10
        self.order_l3 = 5
        self.l1_spread = 2
        self.l2_spread = self.l1_spread * 2
        self.l3_spread = self.l1_spread * 3
        self.order_ids = {}
        self.open_orders = {}
        for contract in self.contracts:
            self.order_ids[contract + ' bid'] = ''
            self.order_ids[contract + ' ask'] = ''
            self.order_ids[contract + ' l1 bid'] = ''
            self.order_ids[contract + ' l1 ask'] = ''
            self.order_ids[contract + ' l2 bid'] = ''
            self.order_ids[contract + ' l2 ask'] = ''
            self.order_ids[contract + ' l3 bid'] = ''
            self.order_ids[contract + ' l3 ask'] = ''
            # self.open_orders[contract] = OpenOrders(contract)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        if success:
            print(f"[DEBUG] Order Cancelled - Order ID: {order_id}")
        else:
            print(f"[DEBUG] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        # for order_key in self.open_orders.keys():
        #     if order_id in self.open_orders[order_key].id_to_qty:
        #         if qty > 0:
        #             self.open_orders[order_key].adjust_qty(order_id, -qty)
        #             self.positions[order_key] += qty
        #             print(f"[DEBUG] Order Fill - {order_key}: +{qty} @ {price}")
        #         else:
        #             self.open_orders[order_key].adjust_qty(order_id, qty)
        #             self.positions[order_key] -= qty
        #             print(f"[DEBUG] Order Fill - {order_key}: {qty} @ {price}")
        #         break
        print("Order Filled")

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print(f"[DEBUG] Order Rejected - Order ID: {order_id}, Reason: {reason}")


    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # print("something was traded")
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        # print("book update")
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # print("Swap response")
        pass


    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        # await self.view_books()
        start_time = datetime.now()
        symbols = ["EPT","DLO","MKU","IGM","BRV"]
        etfs = ["SCP", "JAK"]
        df = pd.read_csv("Case1_Historical.csv")
        predictors = [Prediction(symbol, df[symbol].to_numpy()) for symbol in symbols]
        while True:

            k = 2
            for pred in predictors:
                order_book = self.order_books[pred.name()] if pred.name() in self.order_books else xchange_client.OrderBook()
                pred.update(order_book)
            predictions = dict((pred.name(), pred.predict(k)) for pred in predictors)
            bids = dict((pred.name(), pred.bid(predictions[pred.name()])) for pred in predictors)
            asks = dict((pred.name(), pred.ask(predictions[pred.name()])) for pred in predictors)
            for symbol, _ in predictions.items():
                await self.place_order(symbol, 1, xchange_client.Side.BUY, int(bids[symbol]))
                await self.place_order(symbol, 1, xchange_client.Side.SELL, int(asks[symbol])) 
                # with open(f"./log/round_data{start_time.date()}-{str(start_time.time())[-6]}.txt", "a") as f:
                #     f.write(f"{symbol}: {int(bids[symbol])}, {int(asks[symbol])}\n")
                print(symbol, int(bids[symbol]), int(asks[symbol]))
            
            # ETF Arbitrage
            for etf in etfs:
                if etf == "SCP":
                    price = (3 * predictions["EPT"] + 3*predictions["IGM"] + 4*predictions["BRV"])/10
                elif etf == "JAK":
                    price = (2 * predictions['EPT'] + 5*predictions['DLO'] + 3*predictions['MKU'])/10
                etf_bids = sorted((k,v) for k, v in self.order_books[etf].bids.items() if k > price)
                etf_asks = sorted((k,v) for k, v in self.order_books[etf].asks.items() if k < price)
                for k,v in etf_bids:
                    await self.place_order(etf, v, xchange_client.Side.SELL, k)
                for k,v in etf_asks:
                    await self.place_order(etf, v, xchange_client.Side.BUY, k)
                
                
            # TODO: implement the fade parameter


            # TODO: implement the selling ladder


            # Viewing Positions
            print("My positions:", self.positions)
            # await asyncio.sleep(1)

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
        # asyncio.create_task(self.view_books())
        await self.connect()


async def main():
    SERVER = 'staging.uchicagotradingcompetition.com:3333' # run on sandbox
    my_client = MyXchangeClient(SERVER,"university_of_chicago_umassamherst","ekans-mew-8133")
    await my_client.start()
    return

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())

    


