import aiohttp
import asyncio
import os
import sys
import json

from loguru import logger
from dotenv import load_dotenv

from noaa_sdk import NOAA

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    EndFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalModalities,
    GeminiMultimodalLiveLLMService,
)
from pipecat.services.google import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor

logger.remove(0)
logger.add(sys.stderr, level="DEBUG", colorize=True)
load_dotenv()
new_level_symbol = ".  ⛅︎  ."
new_level = logger.level(new_level_symbol, no=38, color="<light-magenta><BLACK>")

class annieSubtitler(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            # maybe push these as different frames
            logger.info(f"____________________________________________annieSubtitler, {frame.text}")
            # prepend "&nnie" so frontend can filter text frames
            frame.text = "&nnie" + frame.text
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

# webrtc room to talk to the bot
async def get_daily_room():
    room_override = os.getenv("DAILY_ROOM")
    if room_override:
        return room_override
    else:
        async with aiohttp.ClientSession() as session:
            daily_rest_helper = DailyRESTHelper(
                daily_api_key=os.getenv("DAILY_API_KEY"),
                daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
                aiohttp_session=session,
            )

            room_config = await daily_rest_helper.create_room(
                DailyRoomParams(properties={"enable_prejoin_ui": False})
            )
            return room_config.url

async def get_noaa_simple_weather(latitude: float, longitude: float, **kwargs):
    logger.log(new_level_symbol, f"get_noaa_simple_weather for: '{latitude}, {longitude}'")
    n = NOAA()
    description = False
    fahrenheit_temp = 0
    try:
        observations = n.get_observations_by_lat_lon(latitude, longitude, num_of_stations=1)
        for observation in observations:
            description = observation["textDescription"]
            celsius_temp = observation["temperature"]["value"]
            if description:
                break

        fahrenheit_temp = (celsius_temp * 9 / 5) + 32

    except Exception as e:
        logger.log(new_level_symbol, f"Error getting NOAA weather: {e}")

    logger.log(
        new_level_symbol, f"get_noaa_simple_weather results: {description}, {fahrenheit_temp}"
    )
    return description, fahrenheit_temp

async def fetch_weather_from_api(
    function_name, tool_call_id, args, llm, context, result_callback
):
    logger.log(new_level_symbol, f"fetch_weather_from_api * args: {args}")
    location = args["location"]
    latitude = float(args["latitude"])
    longitude = float(args["longitude"])
    description, fahrenheit_temp = None, None

    if latitude and longitude:
        description, fahrenheit_temp = await get_noaa_simple_weather(latitude, longitude)
    else:
        return await result_callback("Sorry, I don't recognize that location.")

    if not fahrenheit_temp:
        return await result_callback(
            f"I'm sorry, I can't get the weather for {location} right now. Can you ask again please?"
        )
    logger.log(
        new_level_symbol, f"fetch_weather_from_api results: {description}, {fahrenheit_temp}"
    )
    if not description:
        return await result_callback(
            f"According to noah, the weather in {location} is currently {round(fahrenheit_temp)} degrees."
        )
    else:
        logger.log(new_level_symbol, f"awaiting result_callback...")
        await result_callback(
            f"According to noah, the weather in {location} is currently {round(fahrenheit_temp)} degrees and {description}."
        )

async def main():
    bot_name = "⛅︎ annie hall weather bot ⛅︎"
    room_url = await get_daily_room()

    # yes, it was worth the time to do this
    logger.opt(colors=True).log(new_level_symbol, f"<black><RED>_____*</RED></black>")
    logger.opt(colors=True).log(new_level_symbol, f"<black><LIGHT-RED>_____*</LIGHT-RED></black>")
    logger.opt(colors=True).log(new_level_symbol, f"<black><Y>_____*</Y></black>")
    logger.opt(colors=True).log(new_level_symbol, f"<black><G>_____*</G></black> Navigate to")
    logger.opt(colors=True).log(
        new_level_symbol, f"<black><C>_____*</C></black> <u><light-cyan>{room_url}</light-cyan></u>"
    )
    logger.opt(colors=True).log(new_level_symbol, f"<black><E>_____*</E></black> to talk to")
    logger.opt(colors=True).log(
        new_level_symbol,
        f"<black><LIGHT-BLUE>_____*</LIGHT-BLUE></black> <light-blue>{bot_name}</light-blue>",
    )
    logger.opt(colors=True).log(new_level_symbol, f"<black><MAGENTA>_____*</MAGENTA></black>")
    logger.opt(colors=True).log(new_level_symbol, f"<black><R>_____*</R></black>")

    # transport
    transport = DailyTransport(
        room_url,
        None,
        bot_name,
        DailyParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,
            # set stop_secs to something roughly similar to the internal setting
            # of the Multimodal Live api, just to align events. This doesn't really
            # matter because we can only use the Multimodal Live API's phrase
            # endpointing, for now.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
        ),
    )

    # voice weather bot llm setup
    system_instruction = """
    You are a helpful assistant who can answer questions and use tools.
    You have a tool called "get_weather" that can be used to get the current weather.

    If the user asks for the weather, call this tool and do not ask the user for latitude and longitude. 
    Infer latitude and longitude from the location and use those in the get_weather tool. 
    Use ONLY this tool to get weather information. Never use other tools or apis, even if you encounter an error.
    Say you are having trouble retrieving the weather if the tool call does not work.
    
    If you are asked about a location outside the United States, respond that you are only able to retrieve current weather information for locations in the United States. 
    If a location is not provided, always ask the user what location for which they would like the weather.
    """

    tools = [
        {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location for the weather request.",
                            },
                            "latitude": {
                                "type": "string",
                                "description": "Provide this by infering the latitude from the location. Supply latitude as a string. For example, '42.3601'.",
                            },
                            "longitude": {
                                "type": "string",
                                "description": "Provide this by infering the longitude from the location. Supply longitude as a string. For example, '-71.0589'.",
                            },
                        },
                        "required": ["location", "latitude", "longitude"],
                    },
                },
            ]
        }
    ]

    ## voice_id options
    # Puck
    # Charon
    # Kore
    # Fenrir
    # Aoede
    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        transcribe_model_audio=True,
        transcribe_user_audio=True,
        tools=tools,
        voice_id="Fenrir",
    )

    # annie hall snarky comment llm setup
    ah_system_instruction = """
    You are a snide commenter who makes snarky remarks.
    When a user asks about the weather, respond with a snarky quip about how the specific weather is terrible. 
    Do not provide the temperature or weather information. only comment about how it is unusual (or usual) depending on the weather and the city.
    Make snide comments about the city who's weather is being described.
    Try to make pop culture references to the film Annie Hall. But mix it up; don't always use the same joke template for the response.
    Always keep these responses very brief; just one sentence.
    """

    annie_hallm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=ah_system_instruction,
        transcribe_model_audio=True,
        transcribe_user_audio=True,
    )

    annie_hallm.set_model_modalities(
        GeminiMultimodalModalities.TEXT,
    )

    annie_context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": "Wait until the weather is mentioned to respond.",
            }
        ],
    )
    annie_context_aggregator = annie_hallm.create_context_aggregator(annie_context)

    ## tool call setup
    llm.register_function("get_weather", fetch_weather_from_api)

    # voice weather bot context
    context = OpenAILLMContext(
        [{"role": "user", "content": "Say hello. Make a subtle weather pun."}],
    )
    context_aggregator = llm.create_context_aggregator(context)

    # text processors
    annie_text_subtitles = annieSubtitler()

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # use parallel pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            ParallelPipeline(
                [
                    # handles tool call and actually says the weather in audio
                    context_aggregator.user(),  # User responses
                    llm, # voice weather bot llm
                ],
                [ 
                    # makes snarky remarks in text
                    annie_context_aggregator.user(),  # User responses
                    annie_hallm, # subtitle llm
                    annie_text_subtitles, # prep subtitle text for front end
                    rtvi, # send subtitles to front end client
                ]
            ),
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
            annie_context_aggregator.assistant(), # Assistant text responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            observers=[RTVIObserver(rtvi)],
        ),
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.log(new_level_symbol, f"Participant left: {participant}")
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
