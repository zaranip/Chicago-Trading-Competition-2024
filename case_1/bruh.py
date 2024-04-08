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
        self.bid_prices = set()
        self.ask_prices = set()

    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj
        if self.id_to_qty[id] == 0:
            self.num_open_orders -= 1
            if id in self.id_to_price:
                price = self.id_to_price[id]
                del self.id_to_price[id]
                if price in self.price_to_id:
                    del self.price_to_id[price]
                    if price in self.bid_prices:
                        self.bid_prices.remove(price)
                    if price in self.ask_prices:
                        self.ask_prices.remove(price)
                del self.id_to_qty[id]

    def modify_order(self, price, qty, old_id, new_id, is_bid):
        if old_id == new_id:
            if old_id not in self.id_to_price:
                self.id_to_price[old_id] = price
                self.price_to_id[price] = old_id
                self.id_to_qty[old_id] = qty
                self.num_open_orders += 1
                if is_bid:
                    self.bid_prices.add(price)
                else:
                    self.ask_prices.add(price)
            else:
                old_price = self.id_to_price[old_id]
                if old_price in self.price_to_id:
                    del self.price_to_id[old_price]
                    if old_price in self.bid_prices:
                        self.bid_prices.remove(old_price)
                    if old_price in self.ask_prices:
                        self.ask_prices.remove(old_price)
                self.price_to_id[price] = old_id
                self.id_to_price[old_id] = price
                self.id_to_qty[old_id] = qty
                if is_bid:
                    self.bid_prices.add(price)
                else:
                    self.ask_prices.add(price)
        else:
            if old_id not in self.id_to_price:
                self.id_to_price[new_id] = price
                self.price_to_id[price] = new_id
                self.id_to_qty[new_id] = qty
                self.num_open_orders += 1
                if is_bid:
                    self.bid_prices.add(price)
                else:
                    self.ask_prices.add(price)
            else:
                del self.price_to_id[self.id_to_price[old_id]]
                del self.id_to_price[old_id]
                del self.id_to_qty[old_id]
                self.price_to_id[price] = new_id
                self.id_to_price[new_id] = price
                self.id_to_qty[new_id] = qty
                if is_bid:
                    self.bid_prices.add(price)
                else:
                    self.ask_prices.add(price)

    def get_qty(self, price):
        p_id = self.price_to_id[price]
        return self.id_to_qty[p_id]

    def get_id(self, price):
        return self.price_to_id[price]

class MarketMakerBot(xchange_client.XChangeClient):
    def __init__(self, host, username, password):
        super().__init__(host, username, password)
        self.contracts = ["EPT", "DLO", "MKU", "IGM", "BRV"]
        self.order_size = 1
        self.edge = 5
        self.fade = 2
        self.levels = 3
        self.level_increment = 5
        self.order_ids = {}
        self.open_orders = {}
        for contract in self.contracts:
            self.order_ids[contract + ' bid'] = ''
            self.order_ids[contract + ' ask'] = ''
            self.open_orders[contract] = OpenOrders(contract)

    async def custom_place_order(self, symbol: str, qty: int, side: Side, px: int = None) -> OrderResponse:
        order_id = await super().place_order(symbol, qty, side, px)
        return OrderResponse(order_id)

    async def place_order(self, contract, qty, side, price, level):
        response = await self.custom_place_order(contract, qty, side, price)
        self.order_ids[contract + f' l{level} {"bid" if side == Side.BUY else "ask"}'] = response.order_id
        self.open_orders[contract].modify_order(price, qty, '', response.order_id, side == Side.BUY)
        print(f"[DEBUG] {contract} - Placed L{level} {'Bid' if side == Side.BUY else 'Ask'} Order. ID: {response.order_id}, Qty: {qty}, Price: {price}")

    async def trade(self):
        print("[DEBUG] Starting Trading Loop")
        while True:
            for contract in self.contracts:
                print(f"[DEBUG] {contract} - Updating Orders")
                book = self.order_books[contract]
                if not book.bids or not book.asks:
                    print(f"[DEBUG] {contract} - Insufficient market data")
                    continue

                best_bid = max(book.bids.keys())
                best_ask = min(book.asks.keys())
                fair_price = (best_bid + best_ask) // 2

                for level in range(self.levels):
                    bid_price = fair_price - self.edge - level * self.level_increment
                    ask_price = fair_price + self.edge + level * self.level_increment

                    if bid_price >= best_bid:
                        bid_price = best_bid - 1  # Penny in
                    if ask_price <= best_ask:
                        ask_price = best_ask + 1  # Penny out

                    bid_qty = self.order_size
                    ask_qty = self.order_size

                    if self.positions[contract] < 0:
                        bid_qty += abs(self.positions[contract]) // self.fade
                    if self.positions[contract] > 0:
                        ask_qty += self.positions[contract] // self.fade

                    if bid_price not in self.open_orders[contract].bid_prices:
                        await self.place_order(contract, bid_qty, Side.BUY, bid_price, level)
                    if ask_price not in self.open_orders[contract].ask_prices:
                        await self.place_order(contract, ask_qty, Side.SELL, ask_price, level)

            await asyncio.sleep(1)

    async def bot_handle_book_update(self, symbol):
        pass

    async def bot_handle_trade_msg(self, symbol, price, qty):
        pass

    async def bot_handle_order_fill(self, order_id, qty, price):
        for contract, open_orders in self.open_orders.items():
            if order_id in open_orders.id_to_qty:
                if qty > 0:
                    open_orders.adjust_qty(order_id, -qty)
                    self.positions[contract] += qty
                    print(f"[DEBUG] Order Fill - {contract}: +{qty} @ {price}")
                else:
                    open_orders.adjust_qty(order_id, qty)
                    self.positions[contract] -= qty
                    print(f"[DEBUG] Order Fill - {contract}: {qty} @ {price}")
                break

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
    bot = MarketMakerBot("staging.uchicagotradingcompetition.com:3333", "university_of_chicago_umassamherst", "ekans-mew-8133")
    await bot.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())