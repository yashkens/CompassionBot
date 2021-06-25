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
import time

from .utils import *
user_histories = {}
user_scores = {}
user_warnings = {}
fight_mode = {}
bot_attacked = {}
fight_stats = {}
achievements = {}
user_name = {}
waiting_for_name = {}
mes_count = {}
ach_count = {}

logger = setup_logger(__name__)


def start_command(update, context):
    """Start a new dialogue when user sends the command "/start"."""

    logger.debug(f"{update.effective_message.chat_id} - User: /start")
    update.message.reply_text("Greetings, Wanderer! "
                              "You successfully escaped the Lair of Charybdis on your quest for the Old Kingdom. "
                              "However, you've destroyed some nearby villages and now locals hate you. "
                              "But such a nuisance won't bother you, will it?\n"
                              "The next goal is the Tower to another dimension. "
                              "At the foot of the Tower there is a shabby-looking wooden door, "
                              "a transparent flickering silhouette seems to guard it. "
                              "You are prepared to meet the ancient magic of the Old Kingdom.\n\n"
                              "_You never know what to expect from emanations of the Old Kingdom._ "
                              "_Try convincing the Keeper to let you in. Or cut your way through it..._\n\n"
                              "_Hit /help for tips_", parse_mode='Markdown')
    replay(update)


def help_command(update, context):
    logger.debug(f"{update.effective_message.chat_id} - User: /help")
    update.message.reply_text("Your task is to make Keeper let you inside the Tower. "
                              "Try different styles: you may befriend it and it will gladly let you in, "
                              "you may threaten it and leave it no other option. "
                              "Figure out what fits your playing style best.\n"
                              "Use /show_scores to see your progress\n\n"
                              "The same applies to the secret mode you might enter :) "
                              "The Old Kingdom curse makes your words less powerful, "
                              "so you need to learn how to resist it. "
                              "Observe. Think what made a good spell.\n\n"
                              "Keeper is a ghost :) "
                              "It may take different forms, that might remind you of various stories of the past.\n\n"
                              "A game by @yashkens, @mjolnika and gpt-2 :)")
                              # parse_mode='Markdown')

# def reset_command(update, context):
#     """Reset the dialogue when user sends the command "/reset"."""
#
#     logger.debug(f"{update.effective_message.chat_id} - User: /stop_game")
#     user_histories[update.effective_message.chat_id] = None
#     user_scores[update.effective_message.chat_id] = 0
#     user_warnings[update.effective_message.chat_id]['had_negative'] = False
#     user_warnings[update.effective_message.chat_id]['had_positive'] = False
#     update.message.reply_text("Beep beep!")


def show_scores_command(update, context):
    logger.debug(f"{update.effective_message.chat_id} - User: /show_scores")
    if update.effective_message.chat_id in user_scores:
        update.message.reply_text("_Your current trust score is: {}_".format(user_scores[update.effective_message.chat_id]),
                                  parse_mode='Markdown')
    else:
        update.message.reply_text("_Start playing first!_",
                                  parse_mode='Markdown')


def fight_stats_command(update, context):
    user_id = update.effective_message.chat_id
    logger.debug(f"{user_id} - User: /show_fight_stats")
    if user_id not in fight_stats:
        update.message.reply_text(
            "_Your current health is 100.\nBot's current health is 100._\n"
            "_You haven't started fighting yet!_",
            parse_mode='Markdown')
        return
    if fight_mode[user_id]:
        update.message.reply_text(
            "_Your current health is: {}_\n"
            "_Bot's current health is: {}_".format(
                fight_stats[user_id]['Your health'], round(fight_stats[user_id]["Ghost's health"], 3)),
            parse_mode='Markdown')
    else:
        update.message.reply_text(
            "_You are out of the fight mode!_",
            parse_mode='Markdown')


def show_achiev_command(update, context):
    user_id = update.effective_message.chat_id
    logger.debug(f"{user_id} - User: /show_achievements")
    ach_text = ''
    if user_id not in achievements:
        update.message.reply_text(
            "_You don't have any achievements yet!_",
            parse_mode='Markdown')
        return
    for label in achievements[user_id]['Labels']:
        if achievements[user_id]['Labels'][label] == 1:
            ach_str = "*{}*\n{}\n\n".format(label, achievements[user_id]['Descriptions'][label])
            ach_text += ach_str
    if not ach_text:
        update.message.reply_text(
            "_You don't have any achievements yet!_",
            parse_mode='Markdown')
    else:
        update.message.reply_text(
            "\U0001F3C6  _Here is the list of your trophies_ \U0001F3C6 \n{}".format(ach_text),
            parse_mode='Markdown')


# def replay(update):
#     user_id = update.effective_message.chat_id
#     if user_id not in user_name:
#         update.message.reply_text("_How can I address you?_", parse_mode='Markdown')
#         waiting_for_name[user_id] = True


def replay(update):
    user_id = update.effective_message.chat_id
    if user_id not in user_name:
        update.message.reply_text("_How can I address you?_", parse_mode='Markdown')
        waiting_for_name[user_id] = True


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

spell_dict = {
    0: '{}: ‧͙⁺˚\*･༓☾`{}`☽༓･\*˚⁺‧͙',
    1: '{}: ☆.｡.:\*`{}`.｡.:\*☆',
    2: '{}: \*＊✿❀`{}`❀✿＊\*',
    3: '{}: \*+:｡.｡`{}`｡.｡:+\*'
}

last_pos_reply = "Embarrassing... My life was lived in vain. I think I can rest in peace now, learning compassion " \
                  "for the first time in my poor life. Bless you, young man."
pos_ending = "_Congratulations! You earned the Bot's trust. Now you can continue the quest!_"
last_pos_action = "_The ghost vanished and left you alone with an eerie feeling._"

last_neg_reply = "May all your kinship burn in hell. Prepare to fight!"
neg_ending = "_Congratulations! Keeper is furious.\np.s. Run..._"


@send_typing_action
def message(self, update, context):
    """Receive message, generate response, and send it back to the user."""

    user_message = update.message.text
    user_id = update.effective_message.chat_id
    logger.debug(f"{update.effective_message.chat_id} - User: {user_message}")
    if user_id not in fight_mode:
        fight_mode[user_id] = False
    if user_id not in achievements:
        achievements[user_id] = generate_achievements()
        ach_count[user_id] = 0
    if user_id not in mes_count:
        mes_count[user_id] = 0

    logger.debug('ach count {}'.format(ach_count[user_id]))
    if ach_count[user_id] == 5:
        achievements[user_id]['Labels']['Cyberpunk 2021'] = 1
        update.message.reply_text(
            "\U0001F3C6 _New achievement!_\n"
            "\U0001F451 *Cyberpunk 2021* \U0001F451\n{}".format(
                achievements[user_id]['Descriptions']['Cyberpunk 2021']),
            parse_mode='Markdown')
        ach_count[user_id] += 1

    # получаем имя, если нужно
    if user_id not in waiting_for_name:
        replay(update)
        return
    if waiting_for_name[user_id]:
        user_name[user_id] = user_message
        waiting_for_name[user_id] = False
        update.message.reply_text("_Now let's start the game!_\n_Start typing something!_", parse_mode='Markdown')
        return

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
            # добавляем ачивки
            achievements[user_id]['Labels']['True Companion Cube'] = 1
            update.message.reply_text(
                "\U0001F3C6  _New achievement!_\n"
                "\U0001F495 *True Companion Cube* \U0001F495\n{}".format(
                    achievements[user_id]['Descriptions']['True Companion Cube']),
                parse_mode='Markdown')
            ach_count[user_id] += 1
            if mes_count[user_id] < 10:
                time.sleep(1)
                achievements[user_id]['Labels']['Munchkin'] = 1
                update.message.reply_text(
                    "\U0001F3C6  _New achievement!_\n"
                    "\U0001F970 *Munchkin* \U0001F970\n{}".format(
                        achievements[user_id]['Descriptions']['Munchkin']),
                    parse_mode='Markdown')
                ach_count[user_id] += 1
            update.message.reply_text(last_pos_reply, parse_mode='Markdown')
            update.message.reply_text(last_pos_action, parse_mode='Markdown')
            time.sleep(1)
            update.message.reply_text(pos_ending, parse_mode='Markdown')
            update.message.reply_text('Restarting the game...', parse_mode='Markdown')
            time.sleep(3)
            user_histories[update.effective_message.chat_id] = None
            user_scores[update.effective_message.chat_id] = 0
            user_warnings[user_id]['had_negative'] = False
            user_warnings[user_id]['had_positive'] = False
            user_name.pop(user_id)
            mes_count[user_id] = 0
            replay(update)
            return
        elif user_scores[user_id] == -5:
            update.message.reply_text(last_neg_reply, parse_mode='Markdown')
            update.message.reply_text(neg_ending, parse_mode='Markdown')
            time.sleep(1)
            user_histories[update.effective_message.chat_id] = None
            user_scores[update.effective_message.chat_id] = 0
            user_warnings[user_id]['had_negative'] = False
            user_warnings[user_id]['had_positive'] = False
            mes_count[user_id] = 0
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
        output_text = replace_vocatives(output_text, user_name[user_id])
        user_histories[user_id] = chat_history_ids

        logger.debug(f"{update.effective_message.chat_id} - Bot: {output_text}")
        # Return response as text
        update.message.reply_text(output_text)
        mes_count[user_id] += 1
        logger.debug(mes_count[user_id])
    else:
        displaybothealth = 100
        realbothealth = 5
        changedhealth = realbothealth
        displayuserhealth = 100
        if user_id not in fight_stats:
            fight_stats[user_id] = defaultdict(int)
        fight_stats[user_id]["Ghost's health"] = displaybothealth
        if "Ghost's changed health" not in fight_stats[user_id]:
            fight_stats[user_id]["Ghost's changed health"] = realbothealth
        if "Your health" not in fight_stats[user_id]:
            fight_stats[user_id]["Your health"] = displayuserhealth
        # fight_stats[user_id]["Your changed health"] = changedhealth
        number = randint(605, 695)

        if number == 666:
            update.message.reply_text('Крибли крабли бумс! Потому что я русский!')
            # добавляем ачивку
            achievements[user_id]['Labels']['Creebly crubly booms'] = 1
            update.message.reply_text(
                "\U0001F3C6  _New achievement!_\n"
                "\U0001F4A5 *Creebly crubly booms* \U0001F4A5\n{}".format(
                    achievements[user_id]['Descriptions']['Creebly crubly booms']),
                parse_mode='Markdown')
        else:
            start_rep = 'I am a ghost. May all your kinship burn in hell. You should die.'
            new_user_input_ids = self.tokenizer.encode(
                start_rep + self.tokenizer.eos_token + user_message + self.tokenizer.eos_token, return_tensors='pt')
            spell = castspell(user_message)
            spell_form = spell_dict[randint(0, 3)]
            scaryspell = spell_form.format(user_name[user_id], spell)
            update.message.reply_text(scaryspell, parse_mode='Markdown')

            fight_stats[user_id]['Last step changed health'] = fight_stats[user_id]["Ghost's health"]
            fight_stats[user_id] = update_stats(fight_stats[user_id], spell, displaybothealth, realbothealth)
            if fight_stats[user_id]["Ghost's changed health"] < 0:
                # добавляем ачивку
                achievements[user_id]['Labels']['No transformer will ever stop me'] = 1
                update.message.reply_text(
                    "\U0001F3C6  _New achievement!_\n"
                    "\U0001F4AA *No transformer will ever stop me* \U0001F4AA\n{}".format(
                        achievements[user_id]['Descriptions']['No transformer will ever stop me']),
                    parse_mode='Markdown')
                ach_count[user_id] += 1
                time.sleep(3)
                update.message.reply_text(
                    "_Congratulations! You defeated the Ghost. Now you may return to your quest and enter the other dimension!_",
                    parse_mode='Markdown')
                time.sleep(1)
                user_histories[user_id] = None
                user_scores[user_id] = 0
                user_warnings[user_id]['had_negative'] = False
                user_warnings[user_id]['had_positive'] = False
                mes_count[user_id] = 0
                fight_mode[user_id] = False
                fight_stats.pop(user_id)
                user_name.pop(user_id)
                update.message.reply_text(
                    "_Restarting the game..._",
                    parse_mode='Markdown')
                time.sleep(3)
                replay(update)
                return

            if fight_mode[user_id]:
                if fight_stats[user_id]["Ghost's health"] < fight_stats[user_id]['Last step changed health']:
                    points = round(fight_stats[user_id]["Ghost's health"], 3)
                    logger.debug(points)
                    logger.error(points)
                    s = "_Keeper lost some health because of your words._ " +\
                        str(points) + " _points left to pass the guard_"
                    logger.debug(s)
                    update.message.reply_text(s,
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
                output_text = replace_vocatives(output_text, user_name[user_id])
                user_histories[user_id] = chat_history_ids
                logger.debug(f"{update.effective_message.chat_id} - Bot: {output_text}")
                update.message.reply_text(output_text)

                if user_id not in bot_attacked:
                    bot_attacked[user_id] = 0
                if bot_attack > 0.35:
                    bot_attacked[user_id] += 1
                    if bot_attacked[user_id] == 4:
                        # добавляем ачивку
                        achievements[user_id]['Labels']['No escape'] = 1
                        update.message.reply_text(
                            "\U0001F3C6 _New achievement!_\n"
                            "\U0001F480 *No escape* \U0001F480\n {}".format(
                                achievements[user_id]['Descriptions']['No escape']),
                            parse_mode='Markdown')
                        ach_count[user_id] += 1
                        time.sleep(3)
                        update.message.reply_text(
                            '_Game over. You became a ghost. Now your fate is to listen eternally to GPT gibberish._',
                            parse_mode='Markdown')
                        time.sleep(1)
                        update.message.reply_text(
                            "_Restarting the game..._",
                            parse_mode='Markdown')
                        fight_mode[user_id] = False
                        user_name.pop(user_id)
                        fight_stats.pop(user_id)
                        replay(update)
                    elif bot_attacked[user_id] == 1:
                        fight_stats[user_id]['Your health'] = randint(81, 99) * displayuserhealth / 100
                        update.message.reply_text(
                            "_Ghost's evil tongue filled you heart with despair. "
                            "Your health dropped to {}._".format(fight_stats[user_id]['Your health']),
                            parse_mode='Markdown')
                    elif bot_attacked[user_id] == 2:
                        fight_stats[user_id]['Your health'] = randint(51, 69) * displayuserhealth / 100
                        update.message.reply_text(
                            "_Ghost's evil tongue filled you heart with despair. "
                            "Your health dropped to {}._".format(fight_stats[user_id]['Your health']),
                            parse_mode='Markdown')
                    elif bot_attacked[user_id] == 3:
                        fight_stats[user_id]['Your health'] = randint(25, 48) * displayuserhealth / 100
                        update.message.reply_text(
                            "_Ghost's evil tongue filled you heart with despair. "
                            "Your health dropped to {}._".format(fight_stats[user_id]['Your health']),
                            parse_mode='Markdown')

            logger.debug('ach count {}'.format(ach_count[user_id]))
            if ach_count[user_id] == 5:
                achievements[user_id]['Labels']['Cyberpunk 2021'] = 1
                update.message.reply_text(
                    "\U0001F3C6 _New achievement!_\n"
                    "\U0001F451 *Cyberpunk 2021* \U0001F451\n{}".format(
                        achievements[user_id]['Descriptions']['Cyberpunk 2021']),
                    parse_mode='Markdown')
                ach_count[user_id] += 1

def error(update, context):
    logger.error(context)
    logger.warning(context.error)


def no_callback(update, context):
    cq = update.callback_query
    user_id = cq.from_user.id
    logger.debug(f'{user_id}: no_callback')
    message = cq.message
    message.reply_text('Coward!')
    time.sleep(3)
    # добавляем ачивку
    achievements[user_id]['Labels']['Run'] = 1
    message.reply_text(
        "\U0001F3C6 _New achievement!_\n"
        "\U0001F921 *Run* \U0001F921\n{}".format(
            achievements[user_id]['Descriptions']['Run']),
        parse_mode='Markdown')
    ach_count[user_id] += 1
    time.sleep(3)
    message.reply_text('_The game is over._\n_Restarting the game..._', parse_mode='Markdown')
    time.sleep(3)
    user_name.pop(user_id)
    if user_id not in user_name:
        message.reply_text("_How can I address you?_", parse_mode='Markdown')
        waiting_for_name[user_id] = True

    if ach_count[user_id] == 5:
        achievements[user_id]['Labels']['Cyberpunk 2021'] = 1
        message.reply_text(
            "\U0001F3C6 _New achievement!_\n"
            "\U0001F451 *Cyberpunk 2021* \U0001F451\n {}".format(
                achievements[user_id]['Descriptions']['Cyberpunk 2021']),
            parse_mode='Markdown')
        ach_count[user_id] += 1
    cq.answer()


def yes_callback(update, context):
    cq = update.callback_query
    user_id = cq.from_user.id
    logger.debug(f'{user_id}: yes_callback')
    message = cq.message
    message.reply_text("⋆ ˚｡⋆୨୧˚You are in fighting mode˚୨୧⋆｡˚ ⋆\n"
                       "Everything you say turns into spells.\n"
                       "Some of them may hurt Keeper's feelings, but you need to figure out the right words first!\n"
                       "Keeper is much more powerful than you, so you need a strategy to win.\n"
                        "Use /show\_fight\_stats to see your current health points.\n\n"
                       "Type your first spell!", parse_mode='Markdown')
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
        dp.add_handler(CommandHandler('help', help_command))
        # dp.add_handler(CommandHandler('stop_game', reset_command))
        dp.add_handler(CommandHandler('show_scores',show_scores_command))
        dp.add_handler(CommandHandler('show_fight_stats', fight_stats_command))
        dp.add_handler(CommandHandler('show_achievements', show_achiev_command))
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
