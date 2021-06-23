from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, PicklePersistence, CallbackQueryHandler
from telegram import ChatAction, InlineKeyboardMarkup, InlineKeyboardButton
from functools import wraps
from urllib.parse import urlencode
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pickle
import os.path
import torch

from .utils import *
user_histories = {}
user_scores = {}
user_warnings = {}
fight_mode = {}
bot_attacked = {}
fight_stats = {}

logger = setup_logger(__name__)


def start_command(update, context):
    """Start a new dialogue when user sends the command "/start"."""

    logger.debug(f"{update.effective_message.chat_id} - User: /start")
    # context.chat_data['turns'] = []
    update.message.reply_text("Just start texting me. "
                              "If I'm getting annoying, type \"/stop_game\". "
                              "Type \"/show_scores\" to see your trust score. "
                              "Make sure to send no more than one message per turn.")


def reset_command(update, context):
    """Reset the dialogue when user sends the command "/reset"."""

    logger.debug(f"{update.effective_message.chat_id} - User: /stop_game")
    user_histories[update.effective_message.chat_id] = None
    user_scores[update.effective_message.chat_id] = 0
    user_warnings[update.effective_message.chat_id]['had_negative'] = False
    user_warnings[update.effective_message.chat_id]['had_positive'] = False
    update.message.reply_text("Beep beep!")


def show_scores_command(update, context):
    logger.debug(f"{update.effective_message.chat_id} - User: /show_scores")
    update.message.reply_text("_Your current trust score is: {}_".format(user_scores[update.effective_message.chat_id]),
                              parse_mode='Markdown')


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    """Retry n times if unsuccessful."""

    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# def translate_message_to_gif(message, **chatbot_params):
#     """Translate message text into a GIF.
#
#     See https://engineering.giphy.com/contextually-aware-search-giphy-gets-work-specific/"""
#
#     params = {
#         'api_key': chatbot_params['giphy_token'],
#         's': message,
#         'weirdness': chatbot_params.get('giphy_weirdness', 5)
#     }
#     url = "http://api.giphy.com/v1/gifs/translate?" + urlencode(params)
#     response = requests_retry_session().get(url)
#     return response.json()['data']['images']['fixed_height']['url']


def self_decorator(self, func):
    """Passes bot object to func command."""

    def command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)

    return command_func


def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        def command_func(self, update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(self, update, context, *args, **kwargs)

        return command_func

    return decorator


send_typing_action = send_action(ChatAction.TYPING)

start_rep_dict = {
    0: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Hundreds of years on the guard! No one shall pass! \\',
    1: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. You shall not pass, so I’d rather you go away and mind your own business. \\',
    2: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. You shall not pass, so I’d rather you go away and mind your own business. \\',
    3: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Even though you are nice to the old ghost, I cannot let you in. You better drink some beer and mourn your good fellow forever stuck in here. \\',
    4: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Even though you are nice to the old ghost, I cannot let you in. You better drink some beer and mourn your good fellow forever stuck in here. \\',
    5: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Please leave this lonely ghost, so I will not grieve my poor life lived in vain. \\',
    6: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Please leave this lonely ghost, so I will not grieve my poor life lived in vain. \\',
    -1: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Go away now or I’ll need to use my powers! \\',
    -2: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Go away now or I’ll need to use my powers! \\',
    -3: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Such a fiend shall vanish. This is your last chance. Leave! \\',
    -4: ' I am a ghost. I served all my life faithfully and keep serving my masters in death. Such a fiend shall vanish. This is your last chance. Leave! \\'
}

last_pos_reply = "Embarrassing... My life was lived in vain. I think I can rest in peace now, learning compassion " \
                  "for the first time in my poor life. Bless you, young man.\n_The ghost vanished and left " \
                  "you with an eerie feeling_"
pos_ending = "_Congratulations! You earned the Bot's trust. Now you can continue the quest!_"
last_pos_action = "_The ghost vanished and left you alone with an eerie feeling._"

last_neg_reply = "May all your kinship burn in hell. Prepare to fight!"
neg_ending = "_Congratulations! You were so annoying that the Bot won't speak to you anymore.\np.s. Run..._"

@send_typing_action
def message(self, update, context):
    """Receive message, generate response, and send it back to the user."""

    user_message = update.message.text
    user_id = update.effective_message.chat_id
    logger.debug(f"{update.effective_message.chat_id} - User: {user_message}")
    if user_id not in fight_mode:
        fight_mode[user_id] = False

    if not fight_mode[user_id]:
        if user_id not in user_scores:
            user_scores[user_id] = 0
        start_rep = start_rep_dict[user_scores[user_id]]
        new_user_input_ids = self.tokenizer.encode(
            start_rep + user_message + self.tokenizer.eos_token, return_tensors='pt')

        # получаем сентимент
        sentiment_info = get_sentiment(user_message)
        # обновляем trust scores
        if user_id not in user_warnings:
            user_warnings[user_id] = {'had_negative': False, 'had_positive': False}
        user_scores[user_id], user_warnings[user_id]['had_negative'], user_warnings[user_id]['had_positive'], mes = \
            update_trust_scores(
            sentiment_info,
            user_scores[user_id],
            user_warnings[user_id]['had_negative'], user_warnings[user_id]['had_positive'])
        if mes:
            update.message.reply_text(mes, parse_mode='Markdown')

        # заканчиваем игру, если пора заканчивать
        if user_scores[user_id] == 7:
            update.message.reply_text(last_pos_reply, parse_mode='Markdown')
            update.message.reply_text(pos_ending, parse_mode='Markdown')
            update.message.reply_text(last_pos_action, parse_mode='Markdown')
            user_histories[update.effective_message.chat_id] = None
            user_scores[update.effective_message.chat_id] = 0
            user_warnings[user_id]['had_negative'] = False
            user_warnings[user_id]['had_positive'] = False
            return
        elif user_scores[user_id] == -1:
            update.message.reply_text(last_neg_reply, parse_mode='Markdown')
            update.message.reply_text(neg_ending, parse_mode='Markdown')
            user_histories[update.effective_message.chat_id] = None
            user_scores[update.effective_message.chat_id] = 0
            user_warnings[user_id]['had_negative'] = False
            user_warnings[user_id]['had_positive'] = False
            custom_keyboard = [[InlineKeyboardButton(text='Yes', callback_data='y'),
                                InlineKeyboardButton(text='No', callback_data='n')]]
            markup = InlineKeyboardMarkup(custom_keyboard, resize_keyboard=True, one_time_keyboard=True)
            update.message.reply_text('Enter the fight?', reply_markup=markup)
            return

        # запоминаем историю текущего юзера
        if user_id not in user_histories:
            user_histories[user_id] = None
        chat_history_ids = user_histories[user_id]

        # готовим историю (последние 10 реплик) + добавляем сентимент
        bot_input_ids = torch.cat([chat_history_ids[:, -10:], new_user_input_ids],
                                  dim=-1) if chat_history_ids is not None else new_user_input_ids
        sentiment_ids = self.tokenizer .encode('\\\\ [{}] || '.format(sentiment_info), return_tensors='pt')
        bot_input_ids = torch.cat([bot_input_ids, sentiment_ids], dim=-1)


        # Generate bot messages
        chat_history_ids = generate_responses(
            bot_input_ids,
            self.generation_pipeline,
            self.tokenizer,
            seed=self.seed,
            debug=self.debug,
            **self.generator_kwargs
        )

        chat_history_ids = chat_history_ids[:, :-2]

        output_text = self.tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        output_text = clean_text(output_text)
        output_text = replace_vocatives(output_text, 'Username')
        user_histories[user_id] = chat_history_ids

        logger.debug(f"{update.effective_message.chat_id} - Bot: {output_text}")
        # Return response as text
        update.message.reply_text(output_text)
    else:
        displaybothealth = 100
        realbothealth = 5
        changedhealth = realbothealth
        displayuserhealth = 100
        if user_id not in fight_stats:
            fight_stats[user_id] = defaultdict(int)
        fight_stats[user_id]["Ghost's health"] = displaybothealth
        fight_stats[user_id]["Ghost's real health"] = realbothealth
        fight_stats[user_id]["Your health"] = displayuserhealth
        fight_stats[user_id]["Your changed health"] = changedhealth
        number = randint(605, 695)

        if number == 666:
            update.message.reply_text('Крибли крабли бумс! Потому что я русский!')
            # ach_dict['Крибли крабли бумс'] = 1
        else:
            start_rep = 'I am a ghost. May all your kinship burn in hell. You should die.'
            new_user_input_ids = self.tokenizer.encode(
                start_rep + self.tokenizer.eos_token + user_message + self.tokenizer.eos_token, return_tensors='pt')
            spell = castspell(user_message)
            # scaryspell = scary_spell(spell)
            scaryspell = '*+:｡.｡{}｡.｡:+*'.format(spell)
            update.message.reply_text(scaryspell)

            last_step_changed_health = fight_stats[user_id]["Your changed health"]
            fight_stats[user_id], fight_stats[user_id]["Your changed health"] = update_stats(
                fight_stats[user_id], spell, fight_stats[user_id]["Ghost's health"], fight_stats[user_id]["Ghost's real health"], fight_stats[user_id]["Your changed health"])
            if fight_stats[user_id]["Your changed health"] < 0:
                update.message.reply_text(
                    "_Congratulations! You defeated the Ghost. Now you may return to your quest and enter the other dimension!_",
                    parse_mode='Markdown')
                # ach_dict['No transformer will ever stop me'] = 1
                # print_achievements(ach_dict, desc_dict)

            if fight_stats[user_id]["Your changed health"] < last_step_changed_health:
                points = fight_stats[user_id]["Ghost's health"]
                update.message.reply_text(f'_Keeper lost some health because of your words. {points} points left to pass the guard_',
                                          parse_mode='Markdown')

            # определим сентимент и прибавим/убавим скоры
            if user_id not in user_histories:
                user_histories[user_id] = None
            chat_history_ids = user_histories[user_id]

            # добвляем новый инпут в историю (пока в ней только 10 последних токенов!)
            bot_input_ids = torch.cat([chat_history_ids[:, -10:], new_user_input_ids],
                                      dim=-1) if chat_history_ids is not None else new_user_input_ids
            sentiment_ids = self.tokenizer.encode('\\\\ [{}] || '.format('negative'), return_tensors='pt')
            bot_input_ids = torch.cat([bot_input_ids, sentiment_ids], dim=-1)

            chat_history_ids = generate_responses(
                bot_input_ids,
                self.generation_pipeline,
                self.tokenizer,
                seed=self.seed,
                debug=self.debug,
                **self.generator_kwargs
            )

            chat_history_ids = chat_history_ids[:, :-2]

            output_text = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            bot_attack = check_insult(output_text)
            output_text = clean_text(output_text)
            output_text = replace_vocatives(output_text, 'Username')
            user_histories[user_id] = chat_history_ids
            logger.debug(f"{update.effective_message.chat_id} - Bot: {output_text}")
            update.message.reply_text(output_text)

            if user_id not in bot_attacked:
                bot_attacked[user_id] = 0
            if bot_attack > 0.35:
                bot_attacked[user_id] += 1
                if bot_attacked[user_id] == 4:
                    update.message.reply_text(
                        '_Game over. You became a ghost. Now your fate is to listen eternally to GPT gibberish._',
                        parse_mode='Markdown')
                    # ach_dict['No escape'] = 1
                    # print_achievements(ach_dict, desc_dict)
                elif bot_attacked[user_id] == 1:
                    fight_stats[user_id]['Your health'] = randint(81, 99) * fight_stats[user_id]["Your health"] / 100
                    update.message.reply_text(
                        "_Ghost's evil tongue filled you heart with despair. "
                        "Your health dropped to {}._".format(fight_stats[user_id]['Your health']),
                        parse_mode='Markdown')
                elif bot_attacked[user_id] == 2:
                    fight_stats[user_id]['Your health'] = randint(51, 69) * fight_stats[user_id]["Your health"] / 100
                    update.message.reply_text(
                        "_Ghost's evil tongue filled you heart with despair. "
                        "Your health dropped to {}._".format(fight_stats[user_id]['Your health']),
                        parse_mode='Markdown')
                elif bot_attacked[user_id] == 3:
                    fight_stats[user_id]['Your health'] = randint(25, 48) * fight_stats[user_id]["Your health"] / 100
                    update.message.reply_text(
                        "_Ghost's evil tongue filled you heart with despair. "
                        "Your health dropped to {}._".format(fight_stats[user_id]['Your health']),
                        parse_mode='Markdown')

def error(update, context):
    logger.warning(context.error)


def no_callback(update, context):
    cq = update.callback_query
    user_id = cq.from_user.id
    logger.debug(f'{user_id}: no_callback')
    message = cq.message
    message.reply_text('Coward!')
    message.reply_text('_The game is over. You can start again._', parse_mode='Markdown')
    cq.answer()


def yes_callback(update, context):
    cq = update.callback_query
    user_id = cq.from_user.id
    logger.debug(f'{user_id}: yes_callback')
    message = cq.message
    message.reply_text("⋆ ˚｡⋆୨୧˚You are in fighting mode˚୨୧⋆｡˚ ⋆ "
                       "Everything you say turns into spells. "
                       "Some of them may hurt Keeper's feelings, but you need to figure out the right words first! "
                       "Keeper is much more powerful than you, so you need a strategy to win.\n"
                        "You can check \stats to see your current health points.", parse_mode='Markdown')
    # "Be cautious! Strange things happen here._You are in fighting mode. Strange things may happen here. Be cautious!_'
    fight_mode[user_id] = True
    cq.answer()


class TelegramBot:
    """Telegram bot based on python-telegram-bot."""

    def __init__(self, **kwargs):
        # Extract parameters
        general_params = kwargs.get('general_params', {})
        device = general_params.get('device', -1)
        seed = general_params.get('seed', None)
        debug = general_params.get('debug', False)

        generation_pipeline_kwargs = kwargs.get('generation_pipeline_kwargs', {})
        generation_pipeline_kwargs = {**{
            'model': 'microsoft/DialoGPT-medium'
        }, **generation_pipeline_kwargs}

        generator_kwargs = kwargs.get('generator_kwargs', {})
        generator_kwargs = {**{
            'max_length': 1000,
            'do_sample': True,
            'clean_up_tokenization_spaces': True
        }, **generator_kwargs}

        prior_ranker_weights = kwargs.get('prior_ranker_weights', {})
        cond_ranker_weights = kwargs.get('cond_ranker_weights', {})

        chatbot_params = kwargs.get('chatbot_params', {})
        if 'telegram_token' not in chatbot_params:
            raise ValueError("Please provide `telegram_token`")
        # if 'giphy_token' not in chatbot_params:
        #     raise ValueError("Please provide `giphy_token`")
        continue_after_restart = chatbot_params.get('continue_after_restart', True)
        data_filename = chatbot_params.get('data_filename', 'bot_data.pkl')

        self.generation_pipeline_kwargs = generation_pipeline_kwargs
        self.generator_kwargs = generator_kwargs
        self.prior_ranker_weights = prior_ranker_weights
        self.cond_ranker_weights = cond_ranker_weights
        self.chatbot_params = chatbot_params
        self.device = device
        self.seed = seed
        self.debug = debug

        # Prepare the pipelines
        self.generation_pipeline, self.tokenizer = load_pipeline('text-generation', device=device, **generation_pipeline_kwargs)
        self.ranker_dict = build_ranker_dict(device=device, **prior_ranker_weights, **cond_ranker_weights)

        # Initialize the chatbot
        logger.info("Initializing the telegram bot...")
        if continue_after_restart:
            persistence = PicklePersistence(data_filename)
            self.updater = Updater(chatbot_params['telegram_token'], use_context=True, persistence=persistence)
            if os.path.isfile(data_filename):
                with open(data_filename, 'rb') as handle:
                    chat_data = pickle.load(handle)['chat_data']
                for chat_id, chat_id_data in chat_data.items():
                    if len(chat_id_data['turns']) > 0:
                        self.updater.bot.send_message(chat_id=chat_id, text="I'm back! Let's resume...")
                    else:
                        self.updater.bot.send_message(chat_id=chat_id, text="I'm live!")
        else:
            self.updater = Updater(chatbot_params['telegram_token'], use_context=True)

        # Add command, message and error handlers
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler('start', start_command))
        dp.add_handler(CommandHandler('stop_game', reset_command))
        dp.add_handler(CommandHandler('show_scores',show_scores_command))
        dp.add_handler(MessageHandler(Filters.text, self_decorator(self, message)))
        dp.add_handler(CallbackQueryHandler(no_callback, pattern='n'))
        dp.add_handler(CallbackQueryHandler(yes_callback, pattern='y'))
        dp.add_error_handler(error)

    def run(self):
        """Run the chatbot."""
        logger.info("Running the telegram bot...")

        # Start the Bot
        self.updater.start_polling()

        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        self.updater.idle()


def run(**kwargs):
    """Run `TelegramBot`."""
    TelegramBot(**kwargs).run()
