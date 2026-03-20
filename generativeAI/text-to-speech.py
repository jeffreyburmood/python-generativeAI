""" this file contains code to perform text-to-speech using the ElevenLabs API
    This example is from their Develoeprs documentation: https://elevenlabs.io/app/developers """

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

client = ElevenLabs(
    api_key="YOUR_API_KEY"
)

audio = client.text_to_speech.convert(
    text="The first move is what sets everything in motion.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)