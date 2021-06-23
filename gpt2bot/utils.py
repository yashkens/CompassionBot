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


sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

insult_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
insult_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")


# labels=[]
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


# def load_pipeline(task, **kwargs):
#     """Load a pipeline."""
#     logger.info(f"Loading the pipeline '{kwargs.get('model')}'...")
#
#     return transformers.pipeline(task, **kwargs)

def load_pipeline(task, **kwargs):
    """Load a pipeline."""
    logger.info(f"Loading the pipeline '{kwargs.get('model')}'...")

    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    model = AutoModelForCausalLM.from_pretrained(kwargs.get('model'))

    return model, tokenizer


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
    """Remove unnecessary spaces."""
    reply = reply.replace('\\', '')
    reply = re.sub('\[((negative)|(positive)|(neutral))\]', '', reply)
    reply = reply.split('||')[0]
    reply = re.sub('^[,.!?$]+', '', reply)  # удаляем пунктуацию в начале предложения
    reply = re.sub(' +([,.!?$]+)', '\g<1>', reply)  # удаляем пробелы перед пунктуацией
    reply = re.sub(',([,.!?$])', '\g<1>', reply)  # супер частный случай - удаляем запятую перед другой пунктуацией
    reply = re.sub(' +', ' ', reply)  # схлопываем пробелы
    reply = reply.lstrip(', .?!').rstrip(', ')
    return reply


def generate_responses(bot_input_ids, pipeline, tokenizer, seed=None, debug=False, **kwargs):
    """Generate responses using a text generation pipeline."""

    # outputs = pipeline(prompt, **kwargs)
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
    # responses = list(map(lambda x: clean_text(x['generated_text'][len(prompt):]), outputs))

    # if debug:
    #     logger.debug(dict(responses=responses))
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
    if len(tokens)>1:
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


def update_stats(stats, text, displaybothealth, realbothealth, changedhealth):
    insult_sc = check_insult(text)
    if insult_sc > 0.55:
        changedhealth = changedhealth - insult_sc
        stats["Ghost's health"] = displaybothealth*round(changedhealth/realbothealth, 2)
    return stats, changedhealth


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


def generate_scores(prompt, responses, pipeline, **kwargs):
    """Generate scores using a text classification pipeline."""
    responses = [prompt + response for response in responses]

    outputs = pipeline(responses, **kwargs)
    return [output['score'] for output in outputs]


def pick_best_response(prompt, responses, ranker_dict, debug=False):
    """Pick the best response according to the weighted average of scores."""
    if len(ranker_dict) == 0:
        return random.choice(responses)

    def _get_wa_group_scores(group_name):
        group_scores = 0
        group_weight_sum = 0
        for model_name, dct in ranker_dict.items():
            if dct['group'] == group_name:
                scores = np.array(generate_scores(
                    prompt,
                    responses,
                    dct['pipeline']
                ))
                if debug:
                    logger.debug(dict(
                        group=group_name,
                        model=model_name,
                        model_scores=scores,
                        model_weight=dct['weight']
                    ))
                group_scores += scores * dct['weight']
                group_weight_sum += dct['weight']
        group_scores /= group_weight_sum
        return group_scores

    group_names = list(map(lambda x: x['group'], ranker_dict.values()))
    if 'prior' in group_names:
        prior_scores = _get_wa_group_scores('prior')
        if debug:
            logger.debug(dict(prior_scores=prior_scores))
    else:
        prior_scores = 1
    if 'cond' in group_names:
        cond_scores = _get_wa_group_scores('cond')
        if debug:
            logger.debug(dict(cond_scores=cond_scores))
    else:
        cond_scores = 1
    final_scores = prior_scores * cond_scores
    if debug:
        logger.debug(dict(final_scores=final_scores))
    return responses[np.argmax(final_scores)]

# python run_bot.py --type=telegram --config=configs/medium-cpu-our.cfg
