from xchangelib import xchange_client
from xchangelib.xchange_client import Side
import asyncio

MAX_ORDER_SIZE = 50
MAX_OPEN_ORDER = 100
MAX_OUTSTANDING_VOLUME = 200
MAX_ABSOLUTE_POSITION = 100

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
                old_price = self.id_to_price[old_id]
                if old_price in self.price_to_id:
                    del self.price_to_id[old_price]
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

class PIPOBot(xchange_client.XChangeClient):
    def __init__(self, host, username, password):
        super().__init__(host, username, password)
        self.contracts = ["EPT", "DLO", "MKU", "IGM", "BRV"]
        self.order_size = 5
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
            self.open_orders[contract] = OpenOrders(contract)

    async def custom_place_order(self, symbol: str, qty: int, side: Side, px: int = None) -> OrderResponse:
        order_id = await super().place_order(symbol, qty, side, px)
        return OrderResponse(order_id)

    async def get_penny_prices(self, contract):
        book = self.order_books[contract]
        if book.asks:
            penny_ask_price = min(book.asks.items(), key=lambda x: x[1] > 0)[0] - 1
        else:
            penny_ask_price = None
        if book.bids:
            penny_bid_price = max(book.bids.items(), key=lambda x: x[1] > 0)[0] + 1
        else:
            penny_bid_price = None
        return penny_bid_price, penny_ask_price

    async def place_penny_orders(self, contract, penny_bid_price, penny_ask_price):
        outstanding_volume = sum(self.open_orders[contract].id_to_qty.values()) if isinstance(self.open_orders[contract], OpenOrders) else 0
        if abs(self.positions[contract]) < MAX_ABSOLUTE_POSITION and self.open_orders[contract].num_open_orders < MAX_OPEN_ORDER and outstanding_volume < MAX_OUTSTANDING_VOLUME:
            old_bid_id = self.order_ids[contract + ' bid']
            old_ask_id = self.order_ids[contract + ' ask']
            bid_response = await self.custom_place_order(contract, min(self.order_size, MAX_ORDER_SIZE), xchange_client.Side.BUY, round(penny_bid_price, 2))
            ask_response = await self.custom_place_order(contract, min(self.order_size, MAX_ORDER_SIZE), xchange_client.Side.SELL, round(penny_ask_price, 2))
            self.order_ids[contract + ' bid'] = bid_response.order_id
            self.open_orders[contract].modify_order(round(penny_bid_price, 2), min(self.order_size, MAX_ORDER_SIZE), old_bid_id, bid_response.order_id)
            self.order_ids[contract + ' ask'] = ask_response.order_id
            self.open_orders[contract].modify_order(round(penny_ask_price, 2), -min(self.order_size, MAX_ORDER_SIZE), old_ask_id, ask_response.order_id)
            print(f"[DEBUG] {contract} - Placed Penny Orders. Bid ID: {bid_response.order_id}, Ask ID: {ask_response.order_id}")

    async def place_level_orders(self, contract, level, spread, penny_bid_price, penny_ask_price):
        outstanding_volume = sum(self.open_orders[contract].id_to_qty.values()) if isinstance(self.open_orders[contract], OpenOrders) else 0
        if abs(self.positions[contract]) < MAX_ABSOLUTE_POSITION and self.open_orders[contract].num_open_orders < MAX_OPEN_ORDER and outstanding_volume < MAX_OUTSTANDING_VOLUME:
            old_bid_id = self.order_ids[contract + f' l{level} bid']
            old_ask_id = self.order_ids[contract + f' l{level} ask']
            bid_response = await self.custom_place_order(contract, min(getattr(self, f'order_l{level}'), MAX_ORDER_SIZE), xchange_client.Side.BUY, round(penny_bid_price - spread, 2))
            ask_response = await self.custom_place_order(contract, min(getattr(self, f'order_l{level}'), MAX_ORDER_SIZE), xchange_client.Side.SELL, round(penny_ask_price + spread, 2))
            self.order_ids[contract + f' l{level} bid'] = bid_response.order_id
            self.open_orders[contract].modify_order(round(penny_bid_price - spread, 2), min(getattr(self, f'order_l{level}'), MAX_ORDER_SIZE), old_bid_id, bid_response.order_id)
            self.order_ids[contract + f' l{level} ask'] = ask_response.order_id
            self.open_orders[contract].modify_order(round(penny_ask_price + spread, 2), -min(getattr(self, f'order_l{level}'), MAX_ORDER_SIZE), old_ask_id, ask_response.order_id)
            print(f"[DEBUG] {contract} - Placed L{level} Orders. Bid ID: {bid_response.order_id}, Ask ID: {ask_response.order_id}")

    async def trade(self):
        print("[DEBUG] Starting Quote Update Loop")
        iteration = 0
        while True:
            for contract in self.contracts:
                print(f"[DEBUG] {contract} - Updating Quotes")
                penny_bid_price, penny_ask_price = await self.get_penny_prices(contract)

                if penny_ask_price is not None and penny_bid_price is not None:
                    print(f"[DEBUG] {contract} - Penny Bid: {penny_bid_price}, Penny Ask: {penny_ask_price}")

                    if penny_ask_price - penny_bid_price > 0:
                        await self.place_penny_orders(contract, penny_bid_price, penny_ask_price)

                        for level in range(1, 4):
                            spread = getattr(self, f'l{level}_spread')
                            if penny_bid_price - spread > 0:
                                await self.place_level_orders(contract, level, spread, penny_bid_price, penny_ask_price)

            iteration += 1
            if iteration % 10 == 0:
                pnl_score = self.calculate_pnl_score()
                print(f"[DEBUG] Current PNL Score: {pnl_score}")

            await asyncio.sleep(1)
    
    def calculate_pnl_score(self):
        pnl = 0
        for contract in self.contracts:
            if self.order_books[contract].bids and self.order_books[contract].asks:
                mid_price = (list(self.order_books[contract].bids.keys())[-1] + list(self.order_books[contract].asks.keys())[0]) / 2
                pnl += self.positions[contract] * mid_price
        pnl += self.positions['cash']
        return pnl

    async def bot_handle_book_update(self, symbol):
        pass

    async def bot_handle_trade_msg(self, symbol, price, qty):
        #print(f"[DEBUG] Trade Message - Symbol: {symbol}, Price: {price}, Quantity: {qty}")
        pass

    async def bot_handle_order_fill(self, order_id, qty, price):
        for order_key in self.open_orders.keys():
            open_orders = self.open_orders[order_key]
            if isinstance(open_orders, OpenOrders):
                if order_id in open_orders.id_to_qty:
                    if qty > 0:
                        open_orders.adjust_qty(order_id, -qty)
                        self.positions[order_key] += qty
                        print(f"[DEBUG] Order Fill - {order_key}: +{qty} @ {price}")
                    else:
                        open_orders.adjust_qty(order_id, qty)
                        self.positions[order_key] -= qty
                        print(f"[DEBUG] Order Fill - {order_key}: {qty} @ {price}")
                    break
            else:
                print(f"[WARNING] self.open_orders[{order_key}] is not an OpenOrders object")

    async def bot_handle_order_rejected(self, order_id, reason):
        print(f"[DEBUG] Order Rejected - Order ID: {order_id}, Reason: {reason}")

    async def bot_handle_cancel_response(self, order_id, success, error):
        if success:
            print(f"[DEBUG] Order Cancelled - Order ID: {order_id}")
        else:
            print(f"[DEBUG] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()


async def main():
    bot = PIPOBot("staging.uchicagotradingcompetition.com:3333", "university_of_chicago_umassamherst", "ekans-mew-8133")
    await bot.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())