# #!/usr/bin/env python3
# """
# Complete Python Telegram Bot Example
# Before running:
# 1. Get your bot token from @BotFather on Telegram
# 2. Install required package: pip install python-telegram-bot
# 3. Replace 'YOUR_BOT_TOKEN' with your actual token
# """

# import logging
# import os
# from datetime import datetime
# from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
# from telegram.ext import (
#     Application, 
#     CommandHandler, 
#     MessageHandler, 
#     CallbackQueryHandler,
#     filters, 
#     ContextTypes
# )

# # Configure logging
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )
# logger = logging.getLogger(__name__)

# # Bot configuration
# BOT_TOKEN = # Replace with your actual token

# # Store user data (in production, use a database)
# user_data = {}

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle /start command"""
#     user = update.effective_user
#     user_id = user.id
    
#     # Store user info
#     user_data[user_id] = {
#         'name': user.first_name,
#         'username': user.username,
#         'started_at': datetime.now()
#     }
    
#     welcome_message = f"""
# 🤖 Hello {user.first_name}! Welcome to your Python bot!

# Available commands:
# /start - Show this welcome message
# /help - Get help information
# /profile - View your profile
# /menu - Show interactive menu
# /echo [message] - Echo your message
# /time - Get current time
# /joke - Get a random joke

# Just send me any message and I'll echo it back!
#     """
    
#     await update.message.reply_text(welcome_message.strip())

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle /help command"""
#     help_text = """
# 🆘 *Bot Help*

# *Basic Commands:*
# • `/start` - Start the bot
# • `/help` - Show this help message
# • `/profile` - View your profile information
# • `/menu` - Show interactive button menu

# *Fun Commands:*
# • `/echo [your message]` - Echo any message
# • `/time` - Get current date and time
# • `/joke` - Get a random programming joke

# *Message Types:*
# • Send any text message - I'll echo it back
# • Send a photo - I'll comment on it
# • Send a document - I'll tell you about it

# *Interactive Features:*
# • Use `/menu` to see inline keyboard buttons
# • Click buttons to interact with the bot

# Need more help? Just ask me anything!
#     """
    
#     await update.message.reply_text(help_text, parse_mode='Markdown')

# async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle /profile command"""
#     user = update.effective_user
#     user_id = user.id
    
#     if user_id in user_data:
#         profile_info = f"""
# 👤 *Your Profile*

# *Name:* {user_data[user_id]['name']}
# *Username:* @{user_data[user_id]['username'] or 'Not set'}
# *User ID:* `{user_id}`
# *Started bot:* {user_data[user_id]['started_at'].strftime('%Y-%m-%d %H:%M:%S')}
# *Total users:* {len(user_data)}
#         """
#     else:
#         profile_info = "Profile not found. Please use /start first."
    
#     await update.message.reply_text(profile_info.strip(), parse_mode='Markdown')

# async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle /menu command with inline keyboard"""
#     keyboard = [
#         [
#             InlineKeyboardButton("🕒 Current Time", callback_data='time'),
#             InlineKeyboardButton("😂 Random Joke", callback_data='joke')
#         ],
#         [
#             InlineKeyboardButton("👤 My Profile", callback_data='profile'),
#             InlineKeyboardButton("📊 Bot Stats", callback_data='stats')
#         ],
#         [
#             InlineKeyboardButton("ℹ️ Help", callback_data='help'),
#             InlineKeyboardButton("🔄 Refresh Menu", callback_data='menu')
#         ]
#     ]
    
#     reply_markup = InlineKeyboardMarkup(keyboard)
    
#     await update.message.reply_text(
#         "🎛️ *Interactive Menu*\n\nChoose an option below:",
#         reply_markup=reply_markup,
#         parse_mode='Markdown'
#     )

# async def echo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle /echo command"""
#     if context.args:
#         message = ' '.join(context.args)
#         await update.message.reply_text(f"🔊 Echo: {message}")
#     else:
#         await update.message.reply_text("Please provide a message to echo. Usage: /echo your message here")

# async def time_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle /time command"""
#     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     await update.message.reply_text(f"🕒 Current time: {current_time}")

# async def joke_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle /joke command"""
#     jokes = [
#         "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
#         "How many programmers does it take to change a light bulb? None, that's a hardware problem! 💡",
#         "Why do Python programmers prefer snakes? Because they don't like Java! ☕🐍",
#         "What's a programmer's favorite hangout place? The Foo Bar! 🍺",
#         "Why did the programmer quit his job? He didn't get arrays! 📊",
#         "How do you comfort a JavaScript bug? You console it! 🖥️"
#     ]
    
#     import random
#     joke = random.choice(jokes)
#     await update.message.reply_text(f"😂 {joke}")

# async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle button clicks from inline keyboards"""
#     query = update.callback_query
#     await query.answer()  # Acknowledge the callback
    
#     user_id = query.from_user.id
    
#     if query.data == 'time':
#         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         await query.edit_message_text(f"🕒 Current time: {current_time}")
    
#     elif query.data == 'joke':
#         jokes = [
#             "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
#             "How many programmers does it take to change a light bulb? None, that's a hardware problem! 💡",
#             "Why do Python programmers prefer snakes? Because they don't like Java! ☕🐍"
#         ]
#         import random
#         joke = random.choice(jokes)
#         await query.edit_message_text(f"😂 {joke}")
    
#     elif query.data == 'profile':
#         if user_id in user_data:
#             profile_info = f"""
# 👤 *Your Profile*

# *Name:* {user_data[user_id]['name']}
# *Username:* @{user_data[user_id]['username'] or 'Not set'}
# *User ID:* `{user_id}`
# *Started bot:* {user_data[user_id]['started_at'].strftime('%Y-%m-%d %H:%M:%S')}
#             """
#             await query.edit_message_text(profile_info.strip(), parse_mode='Markdown')
#         else:
#             await query.edit_message_text("Profile not found. Please use /start first.")
    
#     elif query.data == 'stats':
#         stats = f"""
# 📊 *Bot Statistics*

# *Total users:* {len(user_data)}
# *Bot uptime:* Running
# *Commands available:* 8
# *Last updated:* {datetime.now().strftime('%H:%M:%S')}
#         """
#         await query.edit_message_text(stats.strip(), parse_mode='Markdown')
    
#     elif query.data == 'help':
#         help_text = "🆘 Use /help for detailed help information."
#         await query.edit_message_text(help_text)
    
#     elif query.data == 'menu':
#         # Recreate the menu
#         keyboard = [
#             [
#                 InlineKeyboardButton("🕒 Current Time", callback_data='time'),
#                 InlineKeyboardButton("😂 Random Joke", callback_data='joke')
#             ],
#             [
#                 InlineKeyboardButton("👤 My Profile", callback_data='profile'),
#                 InlineKeyboardButton("📊 Bot Stats", callback_data='stats')
#             ],
#             [
#                 InlineKeyboardButton("ℹ️ Help", callback_data='help'),
#                 InlineKeyboardButton("🔄 Refresh Menu", callback_data='menu')
#             ]
#         ]
#         reply_markup = InlineKeyboardMarkup(keyboard)
#         await query.edit_message_text(
#             "🎛️ *Interactive Menu*\n\nChoose an option below:",
#             reply_markup=reply_markup,
#             parse_mode='Markdown'
#         )

# async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle regular text messages"""
#     message_text = update.message.text
#     user_name = update.effective_user.first_name
    
#     # Simple responses based on keywords
#     if any(word in message_text.lower() for word in ['hello', 'hi', 'hey']):
#         await update.message.reply_text(f"Hello {user_name}! 👋 How are you doing?")
#     elif any(word in message_text.lower() for word in ['how are you', 'how do you do']):
#         await update.message.reply_text("I'm doing great! Thanks for asking. 😊")
#     elif any(word in message_text.lower() for word in ['thanks', 'thank you']):
#         await update.message.reply_text("You're welcome! Happy to help! 🤗")
#     elif 'python' in message_text.lower():
#         await update.message.reply_text("🐍 Python is awesome! I'm built with Python too!")
#     else:
#         # Default echo behavior
#         await update.message.reply_text(f"You said: \"{message_text}\" 💬")

# async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle photo messages"""
#     await update.message.reply_text("📸 Nice photo! I can see you sent me an image.")

# async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle document messages"""
#     file_name = update.message.document.file_name
#     file_size = update.message.document.file_size
#     await update.message.reply_text(
#         f"📄 I received a document!\n"
#         f"*File name:* {file_name}\n"
#         f"*File size:* {file_size} bytes",
#         parse_mode='Markdown'
#     )

# async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle errors"""
#     logger.warning(f'Update {update} caused error {context.error}')

# def main() -> None:
#     """Start the bot"""
#     print("🤖 Starting Python Telegram Bot...")
    
#     # Validate token
#     if BOT_TOKEN == 'YOUR_BOT_TOKEN':
#         print("❌ Error: Please replace 'YOUR_BOT_TOKEN' with your actual bot token!")
#         print("   Get your token from @BotFather on Telegram")
#         return
    
#     # Create application
#     application = Application.builder().token(BOT_TOKEN).build()
    
#     # Add command handlers
#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(CommandHandler("help", help_command))
#     application.add_handler(CommandHandler("profile", profile_command))
#     application.add_handler(CommandHandler("menu", menu_command))
#     application.add_handler(CommandHandler("echo", echo_command))
#     application.add_handler(CommandHandler("time", time_command))
#     application.add_handler(CommandHandler("joke", joke_command))
    
#     # Add callback query handler for inline keyboards
#     application.add_handler(CallbackQueryHandler(handle_callback_query))
    
#     # Add message handlers
#     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
#     application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
#     application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
#     # Add error handler
#     application.add_error_handler(error_handler)
    
#     # Start the bot
#     print("✅ Bot is running! Press Ctrl+C to stop.")
#     application.run_polling(allowed_updates=Update.ALL_TYPES)

# if __name__ == '__main__':
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n🛑 Bot stopped by user")
#     except Exception as e:
#         print(f"❌ Error starting bot: {e}")
