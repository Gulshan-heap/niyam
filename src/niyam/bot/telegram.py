"""Telegram bot: `python -m niyam.bot.telegram` (needs TELEGRAM_BOT_TOKEN).

/ask <question>       answer with sources (plain messages are treated as /ask too)
/follow <id|topic>    alert me when an RBI document changes, or a new one on a topic
/unfollow <id|topic>
/following
"""

import asyncio
import logging

from sqlalchemy.orm import Session
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from niyam.agent.graph import run_agent
from niyam.bot.alerts import follow, following, unfollow
from niyam.bot.send import format_answer
from niyam.config import get_settings
from niyam.db.session import get_engine
from niyam.rag.llm import LiteLLM, grader_llm
from niyam.retrieval.embeddings import get_embedder

log = logging.getLogger("niyam.bot")

HELP = (
    "I answer questions about RBI rules, with the exact source quoted, and can alert you "
    "when a rule changes.\n\n"
    "Just send a question, e.g. <i>What is the minimum cooling-off period for digital loans?</i> "
    "Add a date to ask about the past: <i>...in March 2024?</i>\n\n"
    "/follow 13156 – alert me when RBI document 13156 is amended or replaced\n"
    "/follow digital lending – alert me about new documents on a topic\n"
    "/unfollow …, /following"
)


def answer_text(question: str) -> str:
    with Session(get_engine()) as session:
        state = run_agent(session, question, LiteLLM(), get_embedder(), grader=grader_llm())
    return format_answer(state["answer"], state["as_of_source"])


async def reply(update: Update, text: str) -> None:
    await update.effective_message.reply_text(
        text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reply(update, HELP)


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args) if context.args else (update.effective_message.text or "")
    if len(question.strip()) < 3:
        await reply(update, "Send me a question about RBI rules.")
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        text = await asyncio.to_thread(answer_text, question.strip())
    except Exception:
        log.exception("answering failed")
        text = "Sorry, something went wrong while answering. Please try again."
    await reply(update, text)


async def follow_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = " ".join(context.args or [])
    with Session(get_engine()) as session:
        try:
            sub, new = follow(session, update.effective_chat.id, arg)
        except ValueError as exc:
            await reply(update, str(exc))
            return
        what = f"RBI document {sub.target}" if sub.kind == "doc" else f"“{sub.target}”"
    await reply(update, f"{'Now following' if new else 'Already following'} {what}.")


async def unfollow_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with Session(get_engine()) as session:
        n = unfollow(session, update.effective_chat.id, " ".join(context.args or []))
    await reply(update, "Stopped following it." if n else "You weren't following that.")


async def following_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with Session(get_engine()) as session:
        subs = following(session, update.effective_chat.id)
        lines = [
            f"• {'document ' + s.target if s.kind == 'doc' else '“' + s.target + '”'}" for s in subs
        ]
    await reply(update, "\n".join(lines) if lines else "You aren't following anything yet.")


def build_app(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("follow", follow_cmd))
    app.add_handler(CommandHandler("unfollow", unfollow_cmd))
    app.add_handler(CommandHandler("following", following_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask))
    return app


def main() -> None:
    logging.basicConfig(
        level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    token = get_settings().telegram_bot_token
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN (from @BotFather) in .env to run the bot.")
    build_app(token).run_polling()


if __name__ == "__main__":
    main()
