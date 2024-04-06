from xchangelib import xchange_client
import asyncio

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

class PIPOBot(xchange_client.XChangeClient):
    def __init__(self, host, username, password):
        super().__init__(host, username, password)
        self.contracts = ["EPT", "DLO", "MKU", "IGM", "BRV"]
        self.order_size = 65
        self.order_l1 = 15
        self.order_l2 = 10
        self.order_l3 = 5
        self.l1_spread = 0.02
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

    async def update_quotes(self):
        print("[DEBUG] Starting Quote Update Loop")
        while True:
            for contract in self.contracts:
                print(f"[DEBUG] {contract} - Updating Quotes")
                book = self.order_books[contract]
                if book.asks:
                    penny_ask_price = min(book.asks.keys()) - 1
                else:
                    penny_ask_price = None
                if book.bids:
                    penny_bid_price = max(book.bids.keys()) + 1
                else:
                    penny_bid_price = None

                if penny_ask_price is not None and penny_bid_price is not None:
                    print(f"[DEBUG] {contract} - Penny Bid: {penny_bid_price}, Penny Ask: {penny_ask_price}")

                    if penny_ask_price - penny_bid_price > 0:
                        old_bid_id = self.order_ids[contract + ' bid']
                        old_ask_id = self.order_ids[contract + ' ask']
                        bid_response = await self.place_order(contract, xchange_client.Side.BUY, self.order_size, round(penny_bid_price, 2))
                        ask_response = await self.place_order(contract, xchange_client.Side.SELL, self.order_size, round(penny_ask_price, 2))
                        self.order_ids[contract + ' bid'] = bid_response.order_id
                        self.open_orders[contract].modify_order(round(penny_bid_price, 2), self.order_size, old_bid_id, bid_response.order_id)
                        self.order_ids[contract + ' ask'] = ask_response.order_id
                        self.open_orders[contract].modify_order(round(penny_ask_price, 2), -self.order_size, old_ask_id, ask_response.order_id)

                        print(f"[DEBUG] {contract} - Placed Penny Orders. Bid ID: {bid_response.order_id}, Ask ID: {ask_response.order_id}")

                    if penny_bid_price - self.l1_spread > 0:
                        old_bid_id = self.order_ids[contract + ' l1 bid']
                        old_ask_id = self.order_ids[contract + ' l1 ask']
                        bid_response = await self.place_order(contract, xchange_client.Side.BUY, self.order_l1, round(penny_bid_price - self.l1_spread, 2))
                        ask_response = await self.place_order(contract, xchange_client.Side.SELL, self.order_l1, round(penny_ask_price + self.l1_spread, 2))
                        self.order_ids[contract + ' l1 bid'] = bid_response.order_id
                        self.open_orders[contract].modify_order(round(penny_bid_price - self.l1_spread, 2), self.order_l1, old_bid_id, bid_response.order_id)
                        self.order_ids[contract + ' l1 ask'] = ask_response.order_id
                        self.open_orders[contract].modify_order(round(penny_ask_price + self.l1_spread, 2), -self.order_l1, old_ask_id, ask_response.order_id)

                        print(f"[DEBUG] {contract} - Placed L1 Orders. Bid ID: {bid_response.order_id}, Ask ID: {ask_response.order_id}")

                    if penny_bid_price - self.l2_spread > 0:
                        old_bid_id = self.order_ids[contract + ' l2 bid']
                        old_ask_id = self.order_ids[contract + ' l2 ask']
                        bid_response = await self.place_order(contract, xchange_client.Side.BUY, self.order_l2, round(penny_bid_price - self.l2_spread, 2))
                        ask_response = await self.place_order(contract, xchange_client.Side.SELL, self.order_l2, round(penny_ask_price + self.l2_spread, 2))
                        self.order_ids[contract + ' l2 bid'] = bid_response.order_id
                        self.open_orders[contract].modify_order(round(penny_bid_price - self.l2_spread, 2), self.order_l2, old_bid_id, bid_response.order_id)
                        self.order_ids[contract + ' l2 ask'] = ask_response.order_id
                        self.open_orders[contract].modify_order(round(penny_ask_price + self.l2_spread, 2), -self.order_l2, old_ask_id, ask_response.order_id)

                        print(f"[DEBUG] {contract} - Placed L2 Orders. Bid ID: {bid_response.order_id}, Ask ID: {ask_response.order_id}")

                    if penny_bid_price - self.l3_spread > 0:
                        old_bid_id = self.order_ids[contract + ' l3 bid']
                        old_ask_id = self.order_ids[contract + ' l3 ask']
                        bid_response = await self.place_order(contract, xchange_client.Side.BUY, self.order_l3, round(penny_bid_price - self.l3_spread, 2))
                        ask_response = await self.place_order(contract, xchange_client.Side.SELL, self.order_l3, round(penny_ask_price + self.l3_spread, 2))
                        self.order_ids[contract + ' l3 bid'] = bid_response.order_id
                        self.open_orders[contract].modify_order(round(penny_bid_price - self.l3_spread, 2), self.order_l3, old_bid_id, bid_response.order_id)
                        self.order_ids[contract + ' l3 ask'] = ask_response.order_id
                        self.open_orders[contract].modify_order(round(penny_ask_price + self.l3_spread, 2), -self.order_l3, old_ask_id, ask_response.order_id)

                        print(f"[DEBUG] {contract} - Placed L3 Orders. Bid ID: {bid_response.order_id}, Ask ID: {ask_response.order_id}")

            await asyncio.sleep(1)

    async def bot_handle_book_update(self, symbol):
        pass

    async def bot_handle_trade_msg(self, symbol, price, qty):
        #print(f"[DEBUG] Trade Message - Symbol: {symbol}, Price: {price}, Quantity: {qty}")
        pass

    async def bot_handle_order_fill(self, order_id, qty, price):
        for order_key in self.open_orders.keys():
            if order_id in self.open_orders[order_key].id_to_qty:
                if qty > 0:
                    self.open_orders[order_key].adjust_qty(order_id, -qty)
                    self.positions[order_key] += qty
                    print(f"[DEBUG] Order Fill - {order_key}: +{qty} @ {price}")
                else:
                    self.open_orders[order_key].adjust_qty(order_id, qty)
                    self.positions[order_key] -= qty
                    print(f"[DEBUG] Order Fill - {order_key}: {qty} @ {price}")
                break

    async def bot_handle_order_rejected(self, order_id, reason):
        print(f"[DEBUG] Order Rejected - Order ID: {order_id}, Reason: {reason}")

    async def bot_handle_cancel_response(self, order_id, success, error):
        if success:
            print(f"[DEBUG] Order Cancelled - Order ID: {order_id}")
        else:
            print(f"[DEBUG] Order Cancellation Failed - Order ID: {order_id}, Error: {error}")

    async def start(self):
        asyncio.create_task(self.update_quotes())
        await self.connect()


async def main():
    bot = PIPOBot("staging.uchicagotradingcompetition.com:3333", "university_of_chicago_umassamherst", "ekans-mew-8133")
    await bot.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())