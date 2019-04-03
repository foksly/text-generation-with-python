"""Basic example for a bot that uses inline keyboards.
# This program is dedicated to the public domain under the CC0 license.
"""
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import ReplyKeyboardMarkup
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, ConversationHandler,
                          CallbackQueryHandler, InlineQueryHandler)

import random
import os

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def start(bot, update):
    reply_keyboard = [['/news', '/fiction']]
    reply_markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)

    update.message.reply_text('Please choose:', reply_markup=reply_markup)


def news(bot, update):
    files = os.listdir('news/')
    file = random.choice(files)
    with open(f'news/{file}', 'r') as f:
        text = f.read()
    bot.send_message(chat_id=update.message.chat_id, text=text)

def fiction(bot, update):
    files = os.listdir('fiction/')
    file = random.choice(files)
    with open(f'fiction/{file}', 'r') as f:
        text = f.read()
    bot.send_message(chat_id=update.message.chat_id, text=text)

def help(bot, update):
    update.message.reply_text("Use /start to test this bot.")


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    # Create the Updater and pass it your bot's token.
    updater = Updater("TOKEN")

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('news', news))
    updater.dispatcher.add_handler(CommandHandler('fiction', fiction))
    updater.dispatcher.add_handler(CommandHandler('help', help))
    updater.dispatcher.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    updater.idle()


if __name__ == '__main__':
    main()