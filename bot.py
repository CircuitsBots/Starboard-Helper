import nltk_ai
import dotenv
import os

from discord.ext import commands
from pretty_help import PrettyHelp

dotenv.load_dotenv()
TOKEN = os.getenv('TOKEN')
PREFIX = commands.when_mentioned_or('sh!')

BOT = commands.Bot(
    command_prefix=PREFIX,
    help_command=PrettyHelp()
)


@BOT.event
async def on_ready():
    print(f"Logged in as {BOT.user.name} in the Support Server!")


@BOT.event
async def on_message(message):
    if message.author.bot:
        return
    response = nltk_ai.response(message.content)
    if response is None:
        return
    await message.channel.send(response)


if __name__ == "__main__":
    BOT.run(TOKEN)
