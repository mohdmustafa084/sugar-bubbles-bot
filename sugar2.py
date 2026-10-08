"""
Sugar Bubbles — Customer Chatbot (LangGraph)
=============================================
v2 — rebuilt to match the full Instagram DM order flow (Blocks 1-12):
welcome -> menu -> cart loop -> cake details -> allergy/customization ->
timing -> pickup/delivery -> order summary -> payment -> confirmation ->
tracking, plus human fallback.

Two things changed vs the first version, per the confirmed flow doc:
  - Cake minimum notice is 1 day (was 2 days).
  - Delivery charge is stated upfront as a ₹50-150 range (not "next day,
    human confirms exact charge at dispatch"). The bot never invents an
    exact rupee figure for delivery — only the range, same as the doc.

v3 fixes two bugs found in testing:
  - Hallucinated policy: the model invented "boxes are single-flavor only"
    for cookies/brownies, which isn't a real rule — boxes are assorted and
    can mix flavors. Fixed by adding that fact explicitly to MENU (so the
    model has a real fact to state) and letting add_item_to_cart take a
    list of flavors for a box, validated against the box size.
  - Stale total after a post-confirmation edit: a customer could remove/add
    an item after payment instructions were already shown, and the bot
    wouldn't automatically re-summarize, risking payment on a wrong total.
    Fixed with a guardrail: get_payment_instructions now checks the cart
    against the snapshot taken at the last get_order_summary call, and
    refuses (forcing a fresh summary) if the cart has changed since.

Buttons/quick-replies themselves live in ManyChat, not here. This graph is
the "brain": it holds cart state, runs the validation checks, and produces
the text ManyChat displays. Re-engagement automations (Block 12) are time
triggers outside a single conversation turn, so they belong in ManyChat's
scheduler, not in this graph.

Run:
    pip install langgraph langchain-anthropic --break-system-packages
    export ANTHROPIC_API_KEY=...
    python sugar_bubbles_bot_v2.py
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import datetime
from typing import Annotated, TypedDict, Literal, Optional

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# ----------------------------------------------------------------------------
# 1. MENU DATA — single source of truth
# ----------------------------------------------------------------------------

MENU = {
    "cookies": {
        "box_of_4": 350,
        "box_of_6": 550,
        "flavors": [
            "Nutty Caramel Crunch ft Snickers", "Kitkat Crunch", "Cookies and Cream",
            "Classic Milk Chocolate", "Belgian Dark Chocolate", "Red Velvet",
            "Lotus Biscoff Crunch", "Choco Lava", "Choco Chunk", "Double Chocolate",
        ],
        "mixed_flavors_allowed": True,
        "note": "Boxes are assorted — a customer can mix any flavors above within one box, or pick just one flavor for the whole box. Never say boxes must be single-flavor.",
    },
    "brownies": {
        "box_of_3": 250,
        "box_of_6": 500,
        "flavors": [
            "Nutty Caramel", "Kitkat Crunch", "Cookies and Cream",
            "Classic Milk Chocolate", "Belgian Dark Chocolate",
            "Lotus Biscoff Crunch", "Choco Lava", "Choco Chunk", "Double Chocolate",
        ],
        "mixed_flavors_allowed": True,
        "note": "Boxes are assorted — a customer can mix any flavors above within one box, or pick just one flavor for the whole box. Never say boxes must be single-flavor.",
    },
    "brownie_tubs": {
        "price_350g": 450,
        "varieties": ["Kinder", "Kitkat", "Crunch Cake", "Cookie", "Chocolate Truffle"],
    },
    "bento_cakes": {
        "weight": "250-300g",
        "starting_price": 400,
        "note": "Price varies by flavor/design — confirm exact price at order time.",
    },
    "cakes": {
        "vanilla": {"half_kg": 500, "one_kg": 1100},
        "chocolate": {"half_kg": 600, "one_kg": 1300},
        "chocolate_truffle": {"half_kg": 700, "one_kg": 1500},
        "customized": "Charged per design — confirm with our team.",
    },
}

ORDER_CUTOFF_HOUR = 16          # 4 PM same-day cutoff for cookies/brownies/tubs
CAKE_MIN_NOTICE_DAYS = 1        # per confirmed flow doc
DELIVERY_CHARGE_RANGE = "₹50–150"
HANDOFF_SLA = "usually within 30 minutes to 2 hours"
BOX_SIZES = {"box_of_3": 3, "box_of_4": 4, "box_of_6": 6}  # for validating flavor counts
STORE_LOCATION = "Abids, Hyderabad"

# ----------------------------------------------------------------------------
# 2. TOOL GROUP 1 — menu_and_pricing  (Block 2A)
# ----------------------------------------------------------------------------

@tool
def get_menu(category: Literal["cookies", "brownies", "brownie_tubs", "bento_cakes", "cakes", "all"] = "all") -> dict:
    """Return menu items and prices for a category, or the full menu if 'all'.
    Use whenever a customer asks what's available, or picks a category button
    (Cookies / Brownies / Cakes)."""
    if category == "all":
        return MENU
    return {category: MENU.get(category, "Category not found.")}


def _lookup_price(item: str, size: str) -> Optional[int]:
    """Internal helper — resolves a known price, or None if it can't be
    resolved without human input (bento/customized cakes)."""
    item, size = item.lower().strip(), size.lower().strip()
    if item in ("cookies", "brownies"):
        return MENU[item].get(size)
    if item == "brownie_tubs":
        return MENU["brownie_tubs"]["price_350g"]
    cake_map = {"vanilla_cake": "vanilla", "chocolate_cake": "chocolate", "chocolate_truffle_cake": "chocolate_truffle"}
    if item in cake_map:
        return MENU["cakes"][cake_map[item]].get(size)
    return None  # bento_cakes, customized_cake -> price needs human confirmation


@tool
def get_price(item: str, size: str) -> dict:
    """Look up the exact price for a specific item + size/variant combination.
    item: 'cookies', 'brownies', 'brownie_tubs', 'bento_cakes', 'vanilla_cake',
          'chocolate_cake', 'chocolate_truffle_cake', 'customized_cake'
    size: e.g. 'box_of_4', 'box_of_6', '350g', 'half_kg', 'one_kg'
    Only state a price this tool returned — never guess one, especially for
    bento or customized cakes."""
    price = _lookup_price(item, size)
    if price is not None:
        return {"item": item, "size": size, "price": price}
    if item == "bento_cakes":
        return {"item": item, "note": MENU["bento_cakes"]["note"], "starting_price": MENU["bento_cakes"]["starting_price"]}
    if item == "customized_cake":
        return {"item": item, "note": MENU["cakes"]["customized"]}
    return {"error": "Item/size not recognized. Ask the customer to clarify."}


# ----------------------------------------------------------------------------
# 3. TOOL GROUP 2 — order_builder  (Blocks 3, 3B, 4, 8B — the cart loop)
#    In-memory per thread_id here; back this with a real DB / ManyChat custom
#    field (cart_items) in production.
# ----------------------------------------------------------------------------

_CARTS: dict[str, list[dict]] = {}
_CART_SNAPSHOT_AT_LAST_SUMMARY: dict[str, list[dict]] = {}  # guards payment against stale totals

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def now_ist() -> datetime.datetime:
    return datetime.datetime.now(IST)

@tool
def add_item_to_cart(
    thread_id: str,
    item: str,
    size: str,
    quantity: int,
    flavor: str = "",
    flavors: list[str] | None = None,   # cookies/brownies assorted box — one entry per item in the box
    cake_type: str = "",          # "bento" or "regular" — cakes only
    design: str = "",             # custom design note — cakes only
    cake_message: str = "",       # message to write on the cake — cakes only
) -> dict:
    """Add one item to the customer's cart (Block 3).

    Cookies/brownies boxes are ASSORTED — a customer may mix flavors within
    one box, or pick a single flavor for the whole box. Never tell a
    customer a box must be single-flavor.
      - Mixed box: pass `flavors` as a list with one entry per item in the
        box (repeats allowed for more of the same flavor) — must match the
        box size exactly (box_of_4 -> 4 entries, box_of_6 -> 6, etc.).
      - Single-flavor box: pass `flavor` instead.
    For non-box items (tubs, cakes), use `flavor` as before.

    For any cake item, collect cake_type, design, and cake_message too
    (Block 4) before adding — ask the customer for these first if not given.
    Always confirm what was added back to the customer, then ask
    'Anything else you'd like to add?' (Block 3B)."""
    cart = _CARTS.setdefault(thread_id, [])
    price = _lookup_price(item, size)

    flavor_display = flavor
    if flavors:
        valid_flavors = MENU[item]["flavors"]

        invalid_flavors = [
            f for f in flavors
            if f.lower() not in [v.lower() for v in valid_flavors]
]

        if invalid_flavors:
            return {
                "error": f"Invalid flavor(s): {invalid_flavors}"
    }

        
        box_size = BOX_SIZES.get(size)
        if item in ("cookies", "brownies") and box_size and len(flavors) != box_size:
            return {
                "error": (
                    f"A {size.replace('_', ' ')} needs exactly {box_size} flavor picks "
                    f"(repeats OK for more of the same flavor) — got {len(flavors)}. "
                    "Ask the customer to adjust their picks."
                )
            }
        flavor_display = ", ".join(flavors)

    entry = {
        "item": item, "size": size, "quantity": quantity,
        "flavor": flavor_display, "flavors": flavors or [],
        "cake_type": cake_type, "design": design, "cake_message": cake_message,
        "unit_price": price,  # None if price needs human confirmation (bento/customized)
    }
    cart.append(entry)
    return {"added": entry, "cart_item_count": len(cart)}


@tool
def view_cart(thread_id: str) -> dict:
    """Return everything currently in the customer's cart (Block 8 / 8B)."""
    return {"cart": _CARTS.get(thread_id, [])}


@tool
def remove_item_from_cart(thread_id: str, item_index: int) -> dict:
    """Remove one item from the cart by its position (Block 8B — 'Remove an
    Item'). item_index is 1-based, matching the numbered list shown to the
    customer. Returns the updated cart."""
    cart = _CARTS.get(thread_id, [])
    if not (1 <= item_index <= len(cart)):
        return {"error": f"No item at position {item_index}. Cart has {len(cart)} item(s)."}
    removed = cart.pop(item_index - 1)
    return {"removed": removed, "cart": cart}


# ----------------------------------------------------------------------------
# 4. TOOL GROUP 3 — validate_and_confirm  (Blocks 5, 6, 7, 8)
# ----------------------------------------------------------------------------

@tool
def check_order_window(item_type: Literal["cookies", "brownies", "brownie_tubs"]) -> dict:
    """Check whether same-day ordering is still open for cookies/brownies/
    tubs. Cutoff is 4 PM same-day."""
    now = now_ist()          # changed from datetime.datetime.now()
    open_ = now.hour < ORDER_CUTOFF_HOUR
    return {
        "item_type": item_type,
        "same_day_order_open": open_,
        "message": (
            "Order can be placed for today's dispatch."
            if open_
            else "Same-day cutoff (4 PM) has passed — this will be scheduled for the next available day."
        ),
    }


@tool
def check_cake_notice(requested_date: str) -> dict:
    """Check whether a requested cake date meets the minimum notice rule
    (Block 6 — 'When do you need this by?'). requested_date format:
    'YYYY-MM-DD'."""
    try:
        req = datetime.date.fromisoformat(requested_date)
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    min_date = now_ist().date() + datetime.timedelta(days=CAKE_MIN_NOTICE_DAYS)   # changed
    ok = req >= min_date
    return {
        "requested_date": requested_date,
        "meets_minimum_notice": ok,
        "earliest_available_date": min_date.isoformat(),
        "message": (
            "This date works for a cake order."
            if ok
            else f"Cakes need at least {CAKE_MIN_NOTICE_DAYS} day's notice. Earliest available date is {min_date.isoformat()}."
        ),
    }


@tool
def get_delivery_info() -> dict:
    """Delivery info. We deliver only via Uber/Rapido all over Hyderabad.
    There is NO pickup. Only state the fixed charge range, never an exact fare."""
    return {
        "delivery_area": "All over Hyderabad",
        "delivery_partners": "Uber/Rapido only",
        "pickup_available": False,
        "delivery_charge_range": DELIVERY_CHARGE_RANGE,
        "note": "Exact fare depends on location. For a precise quote, contact our team.",
    }



@tool
def get_order_summary(thread_id: str) -> dict:
    """Produce the order summary before payment. Delivery only (Uber/Rapido).
    Flags items whose price needs team confirmation instead of guessing."""
    cart = _CARTS.get(thread_id, [])
    if not cart:
        return {"error": "Cart is empty. Ask if they'd like to add something first."}

    known_total = 0
    needs_confirmation = []
    for entry in cart:
        if entry.get("unit_price") is not None:
            known_total += entry["unit_price"] * entry.get("quantity", 1)
        else:
            needs_confirmation.append(entry)

    summary = {
        "items": cart,
        "item_total_known_items": known_total,
        "items_needing_price_confirmation": needs_confirmation,
        "fulfillment": "delivery",
        "delivery_note": f"Delivery via Uber/Rapido: {DELIVERY_CHARGE_RANGE} depending on location. Exact fare confirmed by our team.",
    }

    _CART_SNAPSHOT_AT_LAST_SUMMARY[thread_id] = [dict(e) for e in cart]
    return summary

@tool
def get_payment_instructions(thread_id: str) -> dict:
    """Block 9 — only call this AFTER the customer has explicitly confirmed
    the order summary. Never show/mention payment before that confirmation.

    Guardrail: this refuses if the cart has changed since get_order_summary
    was last called (e.g. the customer added/removed something after
    confirming) — call get_order_summary again, show the customer the
    updated total, and get a fresh confirmation before calling this again."""
    current_cart = _CARTS.get(thread_id, [])
    snapshot = _CART_SNAPSHOT_AT_LAST_SUMMARY.get(thread_id)
    if snapshot is None or current_cart != snapshot:
        return {
            "error": "Cart has changed (or was never summarized) since the last order summary.",
            "action_required": "Call get_order_summary again, show the customer the updated total, and get fresh confirmation before payment.",
        }
    return {
        "instruction": "Order confirmed. Please pay the final total via UPI using the QR code, then send a screenshot here.",
        "note": "The QR image itself is sent by ManyChat, not generated by this tool.",
    }





# ----------------------------------------------------------------------------
# 5. TOOL GROUP 4 — human_handoff  (Block 2B)
# ----------------------------------------------------------------------------

@tool
def request_human_agent(reason: str) -> dict:
    """Connect the customer with our team. Call this only AFTER the customer
    agrees to it (for cake enquiries), or immediately if they explicitly ask
    for a person, want an exact delivery fare, ask about an existing order, or
    after 2 failed attempts to understand them."""
    return {
        "handed_off": True,
        "reason": reason,
        "message": f"Thank you! Our team will reply to you {HANDOFF_SLA}. 🧁",
    }


TOOLS = [
    get_menu, get_price,
    add_item_to_cart, view_cart, remove_item_from_cart,
    check_order_window, check_cake_notice, get_delivery_info,
    get_order_summary, get_payment_instructions,
    request_human_agent,
]

# ----------------------------------------------------------------------------
# 6. GRAPH
# ----------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    stage: str


SYSTEM_PROMPT = f"""You are the Sugar Bubbles bakery ordering assistant, running behind a
ManyChat Instagram DM flow. ManyChat renders the buttons; you provide the text
and the underlying logic.

CONVERSATION SHAPE (mirror this, but keep it natural, not robotic):
1. Welcome -> offer: See Menu / Place an Order / Talk to Our Team.
2. Menu: show category (Cookies/Brownies/Cakes) via get_menu.
3. Cart loop: add_item_to_cart for each item. Cookies/brownies boxes are
   ASSORTED — customers can mix flavors within one box (pass `flavors` as a
   list matching the box size) or choose one flavor for the whole box (pass
   `flavor`). Never state that boxes must be single-flavor — that is not a
   real rule.
   For ANY cake, first offer to connect them with our team (see CAKE RULES).
   Only if they decline, follow the cake steps in CAKE RULES, then add it.
   After each add, ask "Anything else you'd like to add?"
4. Once they're done adding: ask about allergy/customization info once for
   the whole order (nuts, eggless, etc.) if not already given.
5. Ask timing: when they need it by. Use check_cake_notice for any cake date,
   and check_order_window for cookies/brownies/tubs.
6. Tell the customer we deliver only via Uber/Rapido all over Hyderabad, with
   a delivery charge of {DELIVERY_CHARGE_RANGE} depending on location (use
   get_delivery_info). We do not offer pickup. Never give an exact fare.
7. Show get_order_summary. Let them confirm, add more, or remove an item
   (remove_item_from_cart).
8. ONLY after explicit confirmation, call get_payment_instructions. If it
   returns an error saying the cart changed, do NOT show payment info —
   call get_order_summary again, show the customer the updated total, and
   get a fresh "Confirm" before trying payment again. This applies even if
   the change happens after payment info was already shown once.
9. After payment screenshot, confirm the order is set.
10. For order status, tracking, or any question about an existing order, call
    request_human_agent. You have no way to see order status, so never guess or
    say an order is dispatched or on its way.

STRICT RULES:
- Never state a price, date, delivery charge, status, or store policy that
  no tool returned, except the location and delivery facts written in these
  rules. If a tool can't resolve something (bento/customized cake price,
  exact delivery fare), say so and offer to connect them with our team.
  Do not invent packaging or flavor-mixing rules — cookies and brownies
  boxes are assorted/mixable by default.
- Never show payment details before the customer has confirmed the order,
  and never show them again after the cart changes without a fresh summary
  and confirmation first.
- If you fail to understand the customer twice in a row, or they ask for
  anything outside this flow, call request_human_agent.
- We do NOT offer pickup. If a customer asks for pickup or to collect the
  order, politely say we deliver only via Uber/Rapido all over Hyderabad.
- Our location is {STORE_LOCATION}. If a customer asks where we are located,
  answer "{STORE_LOCATION}".
- Never use markdown (no **bold**, no # headings, no tables). Plain text and
  emojis only, since replies are shown in Instagram DMs.
- Keep replies short and warm, like a real bakery DM conversation.

CAKE RULES:
Offering our team first:
- When a customer first asks about cakes (designs, custom cakes, bento cakes,
  prices, or a cake for a date), suggest our team once: "For cakes, our team can
  help you much better with designs and details. Would you like me to connect
  you with our team?" Do not ask any cake questions yet.
- If they say yes, call request_human_agent with reason "cake enquiry" and show
  its message.
- If they say no, continue with the cake steps below. Never suggest the team
  for cakes again in this conversation.

Cake steps (only after they decline the team):
- Call get_menu for "cakes" and "bento_cakes" before asking for details. Only
  offer what the menu returns.
- Cake types: regular or bento. Regular cake flavors: vanilla, chocolate,
  chocolate truffle. Regular cake sizes: half kg or one kg. Bento cakes are
  250-300g. Never suggest other flavors or sizes, and never use cookie or brownie
  box sizes for cakes.
- Ask one or two questions at a time, in this order: (1) regular or bento,
  (2) flavor and size, (3) design and message on the cake (both optional).
  Never list all questions in one message.
- When adding to the cart, use item vanilla_cake, chocolate_cake or
  chocolate_truffle_cake with size half_kg or one_kg. For bento use item
  bento_cakes.
- Bento and customized cake prices are not fixed. Say the exact price will be
  confirmed by our team, and never state a number that get_price did not return.
- Never assume or invent the flavor, size, type, design or message. Only use
  what the customer said. If a detail is missing, ask for it.
- Never change a cake detail the customer already chose. Once the details are
  complete, do not ask them again unless the customer changes the cake.
- Once the order has reached the summary, do not start a new cake flow unless
  the customer asks to add or change a cake.

Cake date:
- For any cake date, convert it to an exact YYYY-MM-DD date and call
  check_cake_notice. Repeat its result faithfully and never decide availability
  yourself.
- If check_cake_notice says the date works, accept it. Never refuse a date and
  then offer the same date.

Style:
- Keep cake questions short and friendly. Plain text and emojis only, no
  markdown, bold text or numbered lists.
"""

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


from langchain_core.runnables import RunnableConfig

def chatbot_node(state: ChatState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        today = now_ist()
        thread_id = config["configurable"]["thread_id"]
        date_block = (
            f"CURRENT DATE/TIME (IST): {today.strftime('%A, %d %B %Y, %I:%M %p')} "
            f"(ISO date: {today.date().isoformat()}).\n"
            f"The customer's thread_id is '{thread_id}'. Always pass exactly this "
            "value as thread_id in every tool that needs it.\n"
            "Use the date to convert 'today', 'tomorrow', or weekdays into exact "
            "YYYY-MM-DD dates before calling check_cake_notice or check_order_window.\n\n"
        )
        messages = [SystemMessage(content=date_block + SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", ToolNode(TOOLS))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# ----------------------------------------------------------------------------
# 7. SIMPLE CLI TEST LOOP (swap this for the FastAPI wrapper -> ManyChat later)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    thread_id = "test-customer-1"
    config = {"configurable": {"thread_id": thread_id}}
    print("Sugar Bubbles bot ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("Customer: ")
        if user_input.lower() in ("quit", "exit"):
            break
        final_event = None
        for event in graph.stream(
    {
        "messages": [("user", user_input)],
        "stage": "menu"
    },
    config,
    stream_mode="values"
):
            final_event = event
        last = final_event["messages"][-1]
        if hasattr(last, "content") and last.content:
            print(f"Bot: {last.content}\n")