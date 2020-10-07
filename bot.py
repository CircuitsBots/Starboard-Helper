import ai
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
    response = ai.get_answer(message.content)
    if response.confidence < 3:
        return
    await message.channel.send(message.author.mention + '\n\n' + str(response))


if __name__ == "__main__":
    BOT.run(TOKEN)