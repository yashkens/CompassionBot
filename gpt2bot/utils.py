import configparser
import logging
import transformers
import numpy as np
import random
from random import choice, randint
import re
import stanza
stanza.download('en')
nlp = stanza.Pipeline('en')
chat_history_ids = None
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from collections import defaultdict

# загружаем модели для сентимент и инсульт анализа
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

insult_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
insult_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")

# скачиваем лейблы для сентимент анализа
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    class ColorCodes:
        grey = "\x1b[38;21m"
        green = "\x1b[1;32m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        blue = "\x1b[1;34m"
        light_blue = "\x1b[1;36m"
        purple = "\x1b[1;35m"
        reset = "\x1b[0m"

    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: ColorCodes.grey + format + ColorCodes.reset,
        logging.INFO: ColorCodes.light_blue + format + ColorCodes.reset,
        logging.WARNING: ColorCodes.yellow + format + ColorCodes.reset,
        logging.ERROR: ColorCodes.red + format + ColorCodes.reset,
        logging.CRITICAL: ColorCodes.bold_red + format + ColorCodes.reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name):
    """Set up logger."""
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


# Set up logging
transformers.logging.set_verbosity_error()

logger = setup_logger(__name__)


def set_seed(seed):
    """Set seed globally."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except:
        pass


def parse_optional_int(config, section, option):
    value = config.get(section, option)
    return int(value) if value is not None else None


def parse_optional_float(config, section, option):
    value = config.get(section, option)
    return float(value) if value is not None else None


def parse_optional_bool(config, section, option):
    value = config.get(section, option)
    return value.lower() in ("yes", "true", "t", "1") if value is not None else None


def parse_optional_int_list(config, section, option):
    value = config.get(section, option)
    return list(map(int, value.replace(' ', '').split(','))) if value is not None else None


def parse_config(config_path):
    """Parse config into a dict."""
    logger.info("Parsing the config...")

    # Read the config
    config = configparser.ConfigParser(allow_no_value=True)
    with open(config_path) as f:
        config.read_file(f)

    return dict(
        general_params=dict(
            device=parse_optional_int(config, 'general_params', 'device'),
            seed=parse_optional_int(config, 'general_params', 'seed'),
            debug=parse_optional_bool(config, 'general_params', 'debug')
        ),
        generation_pipeline_kwargs=dict(
            model=config.get('generation_pipeline_kwargs', 'model'),
            config=config.get('generation_pipeline_kwargs', 'config'),
            tokenizer=config.get('generation_pipeline_kwargs', 'tokenizer'),
            framework=config.get('generation_pipeline_kwargs', 'framework')
        ),
        generator_kwargs=dict(
            max_length=parse_optional_int(config, 'generator_kwargs', 'max_length'),
            min_length=parse_optional_int(config, 'generator_kwargs', 'min_length'),
            do_sample=parse_optional_bool(config, 'generator_kwargs', 'do_sample'),
            early_stopping=parse_optional_bool(config, 'generator_kwargs', 'early_stopping'),
            num_beams=parse_optional_int(config, 'generator_kwargs', 'num_beams'),
            num_beam_groups=parse_optional_int(config, 'generator_kwargs', 'num_beam_groups'),
            diversity_penalty=parse_optional_float(config, 'generator_kwargs', 'diversity_penalty'),
            temperature=parse_optional_float(config, 'generator_kwargs', 'temperature'),
            top_k=parse_optional_int(config, 'generator_kwargs', 'top_k'),
            top_p=parse_optional_float(config, 'generator_kwargs', 'top_p'),
            repetition_penalty=parse_optional_float(config, 'generator_kwargs', 'repetition_penalty'),
            length_penalty=parse_optional_float(config, 'generator_kwargs', 'length_penalty'),
            no_repeat_ngram_size=parse_optional_int(config, 'generator_kwargs', 'no_repeat_ngram_size'),
            pad_token_id=parse_optional_int(config, 'generator_kwargs', 'pad_token_id'),
            bos_token_id=parse_optional_int(config, 'generator_kwargs', 'bos_token_id'),
            eos_token_id=parse_optional_int(config, 'generator_kwargs', 'eos_token_id'),
            bad_words_ids=parse_optional_int_list(config, 'generator_kwargs', 'bad_words_ids'),
            num_return_sequences=parse_optional_int(config, 'generator_kwargs', 'num_return_sequences'),
            decoder_start_token_id=parse_optional_int(config, 'generator_kwargs', 'decoder_start_token_id'),
            use_cache=parse_optional_bool(config, 'generator_kwargs', 'use_cache'),
            clean_up_tokenization_spaces=parse_optional_bool(config, 'generator_kwargs', 'clean_up_tokenization_spaces')
        ),
        prior_ranker_weights=dict(
            human_vs_rand_weight=parse_optional_float(config, 'prior_ranker_weights', 'human_vs_rand_weight'),
            human_vs_machine_weight=parse_optional_float(config, 'prior_ranker_weights', 'human_vs_machine_weight')
        ),
        cond_ranker_weights=dict(
            updown_weight=parse_optional_float(config, 'cond_ranker_weights', 'updown_weight'),
            depth_weight=parse_optional_float(config, 'cond_ranker_weights', 'depth_weight'),
            width_weight=parse_optional_float(config, 'cond_ranker_weights', 'width_weight')
        ),
        chatbot_params=dict(
            max_turns_history=parse_optional_int(config, 'chatbot_params', 'max_turns_history'),
            telegram_token=config.get('chatbot_params', 'telegram_token'),
            giphy_token=config.get('chatbot_params', 'giphy_token'),
            giphy_prob=parse_optional_float(config, 'chatbot_params', 'giphy_prob'),
            giphy_max_words=parse_optional_int(config, 'chatbot_params', 'giphy_max_words'),
            giphy_weirdness=parse_optional_int(config, 'chatbot_params', 'giphy_weirdness'),
            continue_after_restart=parse_optional_bool(config, 'chatbot_params', 'continue_after_restart'),
            data_filename=config.get('chatbot_params', 'data_filename')
        )
    )


def load_pipeline(task, **kwargs):
    """Load a pipeline."""
    logger.info(f"Loading the pipeline '{kwargs.get('model')}'...")

    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    model = AutoModelForCausalLM.from_pretrained(kwargs.get('model'))

    return model, tokenizer


# функция для определения и замены вокативов
def replace_vocatives(sentence, username):
  doc = nlp(sentence)
  sent_vocs = []
  for sent in doc.sentences:
    deps = sent.to_dict()
    voc_head_num = -1
    vocative_places = []
    for i, dep in enumerate(deps):
      if dep['deprel'] == 'vocative':
        voc_head_num = dep['id']
        place = 'start_char=' + str(dep['start_char']) + '|' + 'end_char=' + str(dep['end_char'])
        vocative_places.append(place)
      elif dep['head'] == voc_head_num and dep['deprel'] == 'flat':
        place = 'start_char=' + str(dep['start_char']) + '|' + 'end_char=' + str(dep['end_char'])
        vocative_places.append(place)
    if vocative_places:
      sent_vocs.append(vocative_places)
  if not sent_vocs:
    return sentence
  for vocative_places in sent_vocs[::-1]:
    if not vocative_places:
      continue
    match = re.search('start_char=(.+)\|', vocative_places[0])
    start_char = int(match.group(1))
    match = re.search('end_char=(.+)$', vocative_places[-1])
    end_char = int(match.group(1))
    sentence = sentence[:start_char] + username + sentence[end_char:]
  return sentence


def clean_text(reply):
    reply = reply.replace('\\', '')
    reply = re.sub('\[((negative)|(positive)|(neutral))\]', '', reply)
    reply = reply.split('||')[0]
    reply = re.sub('^[,.!?$]+', '', reply)  # удаляем пунктуацию в начале предложения
    reply = re.sub(' +([,.!?$]+)', '\g<1>', reply)  # удаляем пробелы перед пунктуацией
    reply = re.sub(',([,.!?$])', '\g<1>', reply)  # супер частный случай - удаляем запятую перед другой пунктуацией
    reply = re.sub(' +', ' ', reply)  # схлопываем пробелы
    reply = reply.lstrip(', .?!').rstrip(', ')
    reply = reply.replace('!,', '!').replace('?,', '?')
    return reply


def generate_responses(bot_input_ids, pipeline, tokenizer, seed=None, debug=False, **kwargs):
    chat_history_ids = pipeline.generate(
        bot_input_ids, max_length=300,
        min_length=20,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8,
        length_penalty=1.1,
    )

    return chat_history_ids


def get_sentiment(text):
  encoded_input = sentiment_tokenizer.encode(text, return_tensors='pt')
  output = sentiment_model(encoded_input)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)
  ans = labels[np.argmax(scores)]
  return ans


def update_trust_scores(sentiment_info, trust_score, had_negative, had_positive):
    mes = ''
    if sentiment_info == 'negative':
      trust_score -= 1
      if not had_negative:
        mes = ("_Your words influence the relationship with the Bot._\n_It didn't like what you said. Be carefull!_")
        had_negative = True
    elif sentiment_info == 'positive':
      trust_score += 1
      if not had_positive:
        mes = ("_Your words influence the relationship with the Bot._\n_It just became a little happier because of what you said._")
        had_positive = True
    return trust_score, had_negative, had_positive, mes


def castspell(user_input):
    duples = ['goody goody', 'palsy walsy', 'helter skelter',
            'easy peesy lemon squeezy', 'flim flam', 'hootchy kootchy',
            'skimble skamble', 'ricky ticky', 'hurly burly', 'rantum scantum',
            'fingle fangle', 'helter skelter', 'riff raff piff paff',
            'flip flop', 'hurdy gurdy', 'mumbo jumbo', 'walky talky',
            'hanky panky', 'hotch potch', 'higgledy piggledy', 'hocus pocus',
            'criss cross', 'whisky frisky', 'borus snorus', 'joukerie cookerie',
            'hangy bangy', 'hitherum ditherum', 'rumpum scrumpum', 'puper duper agent cooper',
            'puper duper i like snooker', 'nitty gritty', 'super duper', 'super duper',
            'easy peesy', 'easy peesy', 'lemon squeezy']
    tokens = user_input.split(' ')
    dup = choice(duples)
    try:
        last_token = re.search('[^0-9]+', tokens[-1]).group(0)
        last_token = re.search('\w+', last_token).group(0)
        postfix = re.split(last_token, tokens[-1])[1]
        dup_postfix = re.search('(?:\w)[oeiua]([^oeiau]+)?$', dup).group(0)
    except AttributeError:
        return dup + ' ' + user_input
    if len(tokens) > 1:
        line = ' '.join(tokens[:-1])
        return f'{dup} {line} {last_token}{dup_postfix}{postfix}'.strip('!')+'!'
    else:
        return f'{dup} {last_token}{dup_postfix}{postfix}'.strip('!')+'!'


def check_insult(text):
    encoded_input = insult_tokenizer.encode(text, return_tensors='pt')
    output = insult_model(encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores[1]


def update_stats(stats, text, displaybothealth, realbothealth):
    insult_sc = check_insult(text)
    if insult_sc > 0.55:
        stats["Ghost's changed health"] = stats["Ghost's changed health"] - insult_sc
        stats["Ghost's health"] = displaybothealth*round(stats["Ghost's changed health"]/realbothealth, 2)
    return stats


def generate_achievements():
    achievements = ['Munchkin',
                    'True Companion Cube',
                    'Run',
                    'No transformer will ever stop me',
                    'No escape',
                    'Creebly crubly booms',
                    'Cyberpunk 2021']
    descriptions = ["You earned Bot's trust in less than 10 turns",
                    "You showed the poor ghost what a true friendship is like",
                    "Generative pre-trained transformer was scarier than any creature you previously met in your bounty hunter life",
                    "You've mastered your language models understanding",
                    "If only text generation could stop...",
                    "крибли крабли бумс!",
                    "True cyberpunk is conversing with GPT and not what you were presented with.\nCongratulations on getting every storyline trophy!"]
    ach_dict = {}
    desc_dict = {}
    for n, ach in enumerate(achievements):
        ach_dict[ach] = 0
        desc_dict[ach] = descriptions[n]
    user_ach_dict = {}
    user_ach_dict['Labels'], user_ach_dict['Descriptions'] = ach_dict, desc_dict
    return user_ach_dict

# это ненужная нам функция, оставшаяся от авторов оригинального кода
# мы не выбираем ответы, а всегда генерируем один
def build_ranker_dict(**kwargs):
    """Build dictionary of ranker weights and pipelines."""
    kwargs = kwargs.copy()
    human_vs_rand_weight = kwargs.pop('human_vs_rand_weight', None)
    human_vs_machine_weight = kwargs.pop('human_vs_machine_weight', None)
    updown_weight = kwargs.pop('updown_weight', None)
    depth_weight = kwargs.pop('depth_weight', None)
    width_weight = kwargs.pop('width_weight', None)

    ranker_dict = dict()
    if human_vs_rand_weight is not None:
        ranker_dict['human_vs_rand'] = dict(
            pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-human-vs-rand', **kwargs),
            weight=human_vs_rand_weight,
            group='prior'
        )
    if human_vs_machine_weight is not None:
        ranker_dict['human_vs_machine'] = dict(
            pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-human-vs-machine', **kwargs),
            weight=human_vs_machine_weight,
            group='prior'
        )
    if updown_weight is not None:
        ranker_dict['updown'] = dict(
            pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-updown', **kwargs),
            weight=updown_weight,
            group='cond'
        )
    if depth_weight is not None:
        ranker_dict['depth'] = dict(
            pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-depth', **kwargs),
            weight=depth_weight,
            group='cond'
        )
    if width_weight is not None:
        ranker_dict['width'] = dict(
            pipeline=load_pipeline('sentiment-analysis', model='microsoft/DialogRPT-width', **kwargs),
            weight=width_weight,
            group='cond'
        )
    return ranker_dict


# это тоже не наше и нам не нужно
def generate_scores(prompt, responses, pipeline, **kwargs):
    """Generate scores using a text classification pipeline."""
    responses = [prompt + response for response in responses]

    outputs = pipeline(responses, **kwargs)
    return [output['score'] for output in outputs]


# python run_bot.py --type=telegram --config=configs/medium-cpu-our.cfg
# python3 run_bot.py --type=telegram --config=configs/medium-cpu-our.cfg
