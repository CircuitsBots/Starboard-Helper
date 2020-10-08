import nltk_ai
import dotenv
import os
import discord

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
    _response = nltk_ai.response(message.content)
    if _response is None:
        return
    _, title, response = _response.split('\n', 2)
    color = discord.Color.green()
    embed = discord.Embed(
        title=title,
        description=response,
        color=color
    )
    embed.set_footer(
        icon_url=message.author.avatar_url,
        text=message.author
    )
    await message.channel.send(embed=embed)


if __name__ == "__main__":
    BOT.run(TOKEN)
