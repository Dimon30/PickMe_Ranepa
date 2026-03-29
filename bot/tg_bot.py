import asyncio
import logging
import os

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from dotenv import load_dotenv

ENV_FILE = os.getenv("ENV_FILE", ".env")
load_dotenv(ENV_FILE)

from app.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

WELCOME_MESSAGE = (
    "👋 Привет! Я — бот приёмной комиссии РАНХиГС.\n\n"
    "Помогу разобраться с вопросами о поступлении:\n"
    "• проходные баллы и места\n"
    "• образовательные программы\n"
    "• документы и сроки\n\n"
    "Задайте ваш вопрос!"
)

ERROR_MESSAGE = (
    "Произошла ошибка при обработке запроса. "
    "Попробуйте ещё раз или обратитесь в приёмную комиссию."
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(WELCOME_MESSAGE)


async def _keep_typing(bot: Bot, chat_id: int, stop_event: asyncio.Event) -> None:
    """Повторяет typing-индикатор каждые 4 секунды до завершения запроса."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id, "typing")
        except Exception:
            pass
        await asyncio.sleep(4)


@dp.message(F.text)
async def text_handler(message: Message):
    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(message.bot, message.chat.id, stop_event))
    try:
        async with httpx.AsyncClient(timeout=160.0) as client:
            resp = await client.post(
                f"{API_URL}/ask",
                json={"text": message.text, "user_id": message.from_user.id},
            )
            resp.raise_for_status()
            data = resp.json()
        await message.answer(data["answer"])
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error from API: %r", e)
        await message.answer(ERROR_MESSAGE)
    except Exception as e:
        logger.error("Unexpected error: %r", e)
        await message.answer(ERROR_MESSAGE)
    finally:
        stop_event.set()
        typing_task.cancel()


async def main():
    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
