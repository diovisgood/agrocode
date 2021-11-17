import os
# Reduce CPU load. Need to perform BEFORE import numpy and some other libraries.
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import gc
import math
import copy
import json
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Sequence, List, Tuple, Union, Dict
import requests
from tqdm import tqdm
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Setup logging
import logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
    datefmt='%y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
)
log = logging.getLogger('agro')

RANDOM_SEED = 2021

"""
# Общая идея

Эта задача по смыслу сходна с задачей Sentiment Analysis.
То есть, когда тексту в соответствие ставится один или несколько классов,
например: (положительный, негативный, нейтральный)

В данном случае: несколько классов может быть присвоено одновременно (MultiLabel Classification)

Я решил, что для этой цели подойдёт архитектура Transformers.
Точнее, её первая половина: TransformerEncoder.

На вход вместо слов подаётся последовательность эмбедингов (Embeddings).
То есть, каждому слову ставится в соответствие точка в N-мерном пространстве.
Обычно N: от 100 до 300.

Для каждого `embedding` добавляем информацию о положении слова в тексте: `PositionalEncoding`.

Далее несколько слоёв TransformerEncoder обрабатывают всю последовательность сразу,
усиляя одни блоки и ослабляя другие, выделяя, таким образом, важную информацию.

Затем обработанная последовательность сравнивается с некими целевыми эмбедингами (Target Embeddings),
которые описывают то или иное заболевание.
При сравнении вся последовательность сливается в некий единый эмбединг, по одному для каждого класса.

Финальный этап, получившийся набор эмбеддингов (фиксированного размера) пропускается через Linear слой,
чтобы создать вероятности для каждого заболевания.
"""


"""
# Словарь Embeddings для русского языка

Для работы нам потребуются готовые `embeddings` для русских слов.

Есть некоторые доступные для скачивания словари на
[RusVectores](https://rusvectores.org/ru/)

Но размер словарей в них: от 150 до 300 тысяч слов, что довольно мало.
Также, не совсем понятны условия их лицензии.

Есть проект ["Наташа"](https://github.com/natasha/navec).
Размер словаря: 500k слов.

Существует также другой интересный проект:
[DeepPavlov](https://docs.deeppavlov.ai/en/0.0.7/intro/pretrained_vectors.html),
содержащий около 1.5 млн. слов.
Его лицензия: **Apache 2.0** - позволяет как свободное, так и коммерческое использование.

С последним я и буду работать.
Нам потребуется скачать весь словарь, размером 4.14Гб, а затем загрузить его в память.
"""


class GloveModel():
    """
    For a given text returns a list of embeddings
    """
    Pat_Split_Text = re.compile(r"[\w']+|[.,!?;]", flags=re.RegexFlag.MULTILINE)
    Unk_Tag: int = -1
    Num_Tag: int = -1
    
    def __init__(self, substitutions: Optional[str] = None, log: Optional[logging.Logger] = None):
        if log is None:
            log = logging.getLogger()
        # Load Glove Model. Download and convert from text to .feather format (which is much faster)
        glove_file_feather = 'ft_native_300_ru_wiki_lenta_lower_case.feather'
        if not os.path.exists(glove_file_feather):
            glove_file_vec = glove_file_feather.rsplit(os.extsep, 1)[0] + '.vec'
            if not os.path.exists(glove_file_vec):
                log.info('Downloading glove model for russia language from DeepPavlov...')
                self.download_file(
                    'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/'
                    'ft_native_300_ru_wiki_lenta_lower_case.vec'
                )
                log.info('Done')
            # Load model from .vec file
            log.info('Loading Glove Model from .vec format...')
            self.glove = self.load_glove_model(glove_file_vec, size=300)
            log.info(f'{len(self.glove)} words loaded!')
            
            log.info('Saving Glove Model to .feather format...')
            self.glove.reset_index().to_feather(glove_file_feather)
        else:
            log.info('Loading Glove Model from .feather format...')
            self.glove = pd.read_feather(glove_file_feather)
            log.info(f'{len(self.glove)} words loaded!')
        
        log.info('Sorting glove dataframe by words...')
        self.glove.sort_values('word', axis=0, ignore_index=True, inplace=True)
        log.info('Done')
        
        self.subs_tab = {}
        if isinstance(substitutions, str):
            for line in substitutions.splitlines():
                words = line.strip().lower().split()
                if len(words) < 2:
                    continue
                self.subs_tab[words[0]] = words[1:]
        log.info(f'Using the substitutions table of {len(self.subs_tab)} records')
        
        """
        Для неизвестных слов я буду использовать embedding слова 'unk'.
        А для чисел - embedding слова 'num'.
        Я не уверен, что авторы DeepPavlov именно так и планировали.
        Но стандартных '<unk>' или '<num>' я там не обнаружил.
        """
        self.Unk_Tag = int(self.glove.word.searchsorted('unk'))
        self.Num_Tag = int(self.glove.word.searchsorted('num'))
        assert self.glove.word[self.Unk_Tag] == 'unk', 'Failed to find "unk" token in Glove'
        assert self.glove.word[self.Num_Tag] == 'num', 'Failed to find "num" token in Glove'
    
    def __len__(self):
        return len(self.glove)
    
    def __getitem__(self, text: str) -> List[np.ndarray]:
        tags = self.text2tags(text, return_offsets=False)
        embeddings = [self.tag2embedding(tag) for tag in tags]
        return embeddings
    
    @staticmethod
    def download_file(url: str, block_size=4096, file_name: Optional[str] = None):
        """Downloads file and saves it to local file, displays progress bar"""
        with requests.get(url, stream=True) as response:
            if file_name is None:
                if 'Content-Disposition' in response.headers.keys():
                    file_name = re.findall('filename=(.+)', response.headers['Content-Disposition'])[0]
            if file_name is None:
                file_name = url.split('/')[-1]
            expected_size_in_bytes = int(response.headers.get('content-length', 0))
            received_size_in_bytes = 0
            with tqdm(total=expected_size_in_bytes, unit='iB', unit_scale=True, position=0, leave=True) as pbar:
                with open(file_name, 'wb') as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        pbar.update(len(data))
                        received_size_in_bytes += len(data)
            if (expected_size_in_bytes != 0) and (expected_size_in_bytes != received_size_in_bytes):
                raise UserWarning(f'Incomplete download: {received_size_in_bytes} of {expected_size_in_bytes}')
    
    @staticmethod
    def load_glove_model(file_name: str, encoding: str = 'utf-8', size: Optional[int] = None) -> pd.DataFrame:
        """
        Loads glove model from text file into pandas DataFrame
        Returns
        -------
        df : pd.DataFrame
            A dataframe with two columns: 'word' and 'embedding'.
            The order of words is preserved as in the source file. Thus it may be unsorted!
        """
        words, embeddings = [], []
        with tqdm(total=os.path.getsize(file_name), unit='iB', unit_scale=True, position=0, leave=True) as pbar:
            with open(file_name, 'r', encoding=encoding) as f:
                first_line = True
                line = f.readline()
                while line:
                    split_line = line.split()
                    line = f.readline()
                    if first_line:
                        first_line = False
                        if len(split_line) == 2:
                            if size is None:
                                size = int(split_line[1])
                            else:
                                assert size == int(split_line[1]), \
                                    f'Size specified at the first line: {int(split_line[1])} does not match: {size}'
                            continue
                    if size is not None:
                        word = ' '.join(split_line[0:-size])
                        embedding = np.array(split_line[-size:], dtype=np.float32)
                        assert len(embedding) == size, f'{line}'
                    else:
                        word = split_line[0]
                        embedding = np.array(split_line[1:], dtype=np.float32)
                        size = len(embedding)
                    words.append(word)
                    embeddings.append(embedding)
                    pbar.update(f.tell() - pbar.n)
        return pd.DataFrame({'word': words, 'embedding': embeddings})
    
    def word2tag(self, word: str, use_unk=True, use_num=True) -> int:
        tag = self.glove.word.searchsorted(word)
        if tag == len(self.glove):
            return self.Unk_Tag if use_unk else -1
        if self.glove.word[tag] == word:
            return int(tag)
        if use_num:
            try:
                num = float(word)
                return self.Num_Tag
            except ValueError:
                pass
        return self.Unk_Tag if use_unk else -1
    
    def tag2embedding(self, tag: int) -> np.ndarray:
        return self.glove.embedding[tag]
    
    def word2embedding(self, word: str) -> np.ndarray:
        tag = self.word2tag(word)
        return self.glove.embedding[tag]
    
    @staticmethod
    def separate_number_chars(s) -> List[str]:
        """
        Does what its name says.
        Examples
        --------
        'october10' -> ['october', '10']
        '123asdad' -> ['123', 'asdad']
        '-12.3kg' -> ['-12.3', 'kg']
        '1aaa2' -> ['1', 'aaa', '2']
        """
        res = re.split(r'([-+]?\d+\.\d+)|([-+]?\d+)', s.strip())
        res_f = [r.strip() for r in res if r is not None and r.strip() != '']
        return res_f
    
    def text2tags(self, text: str, return_offsets=True) -> Union[List[int], Tuple[List[int], List[int]]]:
        text = text.lower()
        tags = []
        offsets = []
        for m in self.Pat_Split_Text.finditer(text):
            # Get next word and its offset in text
            word = m.group(0)
            offset = m.start(0)
            # Current word can be converted to a list of words due to substitutions: 'Iam' -> ['I', 'am']
            # or numbers and letters separations: '123kg' -> ['123', 'kg']
            if word in self.subs_tab:
                words = self.subs_tab[word]
            else:
                words = self.separate_number_chars(word)
            # Get a list of tags, generated on the source word.
            # Note: they all point to the same offset in the original text.
            for word in words:
                tags.append(self.word2tag(word))
                offsets.append(offset)
        if not return_offsets:
            return tags
        return tags, offsets


"""
# Решение проблемы отсутствующих слов
По условиям конкурса:
> Запрещается Использовать ручную *разметку* *тестовых* данных в качестве решения, в т.ч. любые сервисы разметки.

При этом, не вполне ясно определено, что подразумевается под *разметкой* данных.
В любом случае, речь в запрете идёт о **тестовых** данных.

Поэтому, условия конкурса НЕ запрещают мне подготовить словарь для исправления некоторых ошибок,
а также для замены некоторых слов, которые отсутствуют в `embeddings`.
"""
SUBSTITUTIONS = """
цинксодержащие цинк содержащие
проглистогонила дала препарат от глистов
проглистогонил дал препарат от глистов
проглистовать дать препарат от глистов
проглистовали дали препарат от глистов
глистогонить дать препарат от глистов
противогельминтные против глистов
спазган обезболивающий препарат
спазгане обезболивающем препарата
спазганом обезболивающим препаратом
чемерицы рвотный препарат
чемерица рвотный препарат
чемерицей рвотным препаратом

седимин железосодерщащий препарат
левомеколь антисептической мазью
левомиколь антисептическая мазь
левомеколью антисептической мазью
левомиколью антисептической мазью
левомеколем антисептической мазью
левомиколем антисептической мазью
пребиотик пробиотик
пребеотик пробиотик
прибиотик пробиотик
прибеотик пробиотик
прибиотика пробиотик
пробиотика пробиотик
прибеотика пробиотик
пробеотика пробиотик

отел отёл
отелл отёл
оттел отёл
оттелл отёл
отелу отёлу
отеллу отёлу
оттелу отёлу
оттеллу отёлу
отёле родах
отёлл отёл
оттёл отёл
оттёлл отёл
отёллу отёлу
оттёлу отёлу
оттёллу отёлу
оттела отёла
отелла отёла
оттелла отёла
оттёла отёла
отёлла отёла
оттёлла отёла
отёлом отелом
оттелом отелом
отеллом отелом
оттеллом отелом
оттёлом отелом
отёллом отелом
оттёллом отелом
отелы отёлы
отеллы отёлы
оттелы отёлы
оттеллы отёлы
отелов отёлов
отеллов отёлов
оттелов отёлов
оттеллов отёлов
телилась рожала
отелилась родила
отёлилась родила

бурёнке корове
буренке корове
тёлке корове
тёлочке корове
тёлочка телочка
тёлочку корову
укоровы у коровы
телке корове
телки коровы
бычёк бычек
телятки телята
первотелка корова
первотелки коровы
новотельной коровы
коровушки коровы

доим дою
доишь дою
сдаиваю дою
выдаиваю дою
сдаиваем дою
выдаивем дою
додаиваю дою до конца
доились давали молоко
доется доится
выдаивании доении
сцеживал доил
сцеживала доила
доением отбором молока
сдаивание дойка

отпоил напоил
отпоила напоила
отпоили напоили
выпоить напоить
выпоили напоили
пропоить напоить
пропоили напоили
поите давайте пить
поили давали пить

свищик свищ
свищики свищи

гноящийся гнойный
выдрана вырвана

апитит аппетит
аппитит аппетит
апиттит аппетит
апетит аппетит
апеттит аппетит
опетит аппетит
оппетит аппетит
опеттит аппетит
оппеттит аппетит
опитит аппетит
зарастёт зарастет
пощаще почаще
паздбища пастбища
причинай причиной
пречинай причиной
килограм килограмм
килаграм килограмм
килаграмм килограмм

пузатенькая пузатая

абсцез абсцесс
абсцес абсцесс
абсцезс абсцесс
абсцэз абсцесс
абсцэс абсцесс
абсцэзс абсцесс

перестраховываюсь чересчур переживаю
непроходили не проходили

обкололи поставили укол
колили кололи
вколото поставлено
вкалол вколол
кольнул уколол
истыкали прокололи

накосячил ошибся
ветаптеке ветеринарной аптеке
ветаптеки ветеринарной аптеки
ветаптеку ветеринарную аптеку

житкостью жидкостью
рацеоне рационе
худющие худые
здох сдох
скаждым с каждым
четветый четвертый
ожёг ожег
поднятся подняться

захромала начала хромать

искривился стал кривым

расцарапывает царапает
вычесывает чешется

подшатываются шатаются
пошатываются шатаются

ветиринар ветеринар
ветеринат ветеринар
ветеренаров ветеринаров
ветиренаров ветеринаров
ветеренара ветеринара
ветиренара ветеринара
ветеренару ветеринару
ветиренару ветеринару
ветеренаром ветеринаром
ветиренаром ветеринаром
ветеренары ветеринары
ветиренары ветеринары

расслоилось разделилось на слои
разслоилось разделилось на слои
дегтеобразное похожее на деготь
дегтеобразная похожая на деготь
кремообразное похожее на крем
кремообразная похожая на крем

волосики волосы
залысина лысина
облазит линяет

уменя у меня
делоть делать
дилоть делать
дилать делать

зади сзади
взади сзади
взаде сзади

какба как-бы
какбы как-бы

прошупывается прощупывается
прашупывается прощупывается
пращупывается прощупывается

клещь клещ
клешь клещ
клеш клещ
клещь клещ
клещем клещ
клешем клещ

рвотная рвотный

тужится напрягается
тужиться напрягаться
какает испражняется
срет испражняется
срёт испражняется
дрищет испражняется
запоносил начал поносить
дристать поносить

подсохло высохло
нарывать опухать

оттекла отекла
отекшее опухшее
отёкшее опухшее
припух опух
припухло опухло
припухла опухла
опухшая набухшая
апухшая набухшая

вздувает раздувает
воспаленное поврежденное
вспухшие опухшие
расперло опухло

зашибла ушибла

припухлостей шишек
припухлостями шишками
припухлостям шишкам
припухлостях шишках
припушлостям шишкам

покраснений красноты

жидковат жидкий
жидковатый жидкий
жидковато жидко
жиденький жидкий

животина животное
животины животного
животине животному
животиной животным
животиною животным

температурит имеет повышенную температуру
темпиратурит имеет повышенную температуру
тимпературит имеет повышенную температуру
тимпиратурит имеет повышенную температуру
температурить иметь повышенную температуру
темпиратурить иметь повышенную температуру
тимпиратурить иметь повышенную температуру
тимпературить иметь повышенную температуру

покашливает кашляет
подкашливает кашляет
покашливают кашляют
подкашливают кашляют
откашливаются кашляют

покашливал кашлял
подкашливал кашлял
покашливали кашляли
подкашливали кашляли
откашливались кашляли
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe: th.Tensor = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: th.Tensor, mask: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Parameters
        ----------
            x: torch.Tensor (sequence_length, batch_size, d_model)
            mask: torch.Tensor (batch_size, sequence_length)
        Returns
        -------
            output: (sequence_length, batch_size, d_model)
        """
        pos = self.pe[:x.size(0), :]
        x = x + th.permute(mask, (1, 0)).unsqueeze(2).expand_as(x) * pos
        return self.dropout(x)


class DoctorText(nn.Module):
    """
    PyTorch Model based on sentiment predictions
    
    It uses only Encoder part of the Transformers architecture
    """
    def __init__(self,
                 glove: GloveModel,
                 d_model: int,
                 initial_targets: Sequence[str],
                 num_heads=8,
                 num_layers=6,
                 d_hidden=1024,
                 max_len=5000,
                 dropout=0.1,
                 causal_mask=True,
                 device: Optional[Union[str, th.device]] = None):
        super().__init__()
        self.glove = glove
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_targets = len(initial_targets)
        self.d_hidden = d_hidden
        self.max_len = max_len
        self.dropout = dropout
        self.causal_mask = causal_mask
        self.device = device
        
        self.position_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=d_hidden, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        """
        Our `targets` also have embeddings. One embedding - if target is described by a word,
        or multiple - if target is described by a phrase.
        We can initialize target embeddings randomly to be trained during training cycle.
        But, maybe, we can speed up the learning process if we initialize target embeddings
        with their `original meaning`.
        We can easily do that by analyzing target description and summing up the
        respective embeddings of the words of that description.
        Of course, anyway, targets will be changed during the training cycle.
        """
        initial_target_embeddings = []
        for target_phrase in initial_targets:
            target_embeddings = self.glove[target_phrase]
            target_embedding = np.mean(np.stack(target_embeddings), axis=0)
            initial_target_embeddings.append(target_embedding)
        initial_target_embeddings = th.tensor(initial_target_embeddings, dtype=th.float32).unsqueeze(dim=1)
        self.targets = nn.Parameter(initial_target_embeddings)

        self.collect = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, bias=False, add_bias_kv=False)
        
        self.output = nn.Linear(self.num_targets * d_model, self.num_targets)
        
    def forward(self, texts: List[str], output_spans=False, threshold=0.7):
        # Convert batch of texts into tensor of embeddings
        x, padding_mask, batch_offsets = self.texts2batch(texts)
        # x has shape: (sequence_length, batch_size, d_model)
        # padding_mask has shape: (batch_size, sequence_length)
        # batch_offsets is the list of length of batch_size, which contains a list of offsets for each tag
        
        # Add positional information into x
        x = self.position_encoder.forward(x, mask=padding_mask)
        
        # Initialize self-attention mask, so that words could attend only prior words.
        attn_mask = None
        if self.causal_mask:
            attn_mask = th.full((len(x), len(x)), -math.inf, device=x.device, dtype=x.dtype)
            attn_mask = th.triu(attn_mask, diagonal=1)

        x = self.transformer_encoder.forward(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        # x still has shape (sequence_length, batch_size, d_model)
        
        # Combine source embeddings into one embedding, one for each target
        attn_output, attn_weights = self.collect.forward(
            query=self.targets.expand((self.num_targets, x.size(1), self.d_model)),
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            need_weights=True
        )
        # attn_output has the shape: (num_targets, batch_size, d_model)
        # attn_weights has the shape: (batch_size, num_targets, sequence_length)
        
        attn_output = attn_output.permute((1, 0, 2)).reshape(x.size(1), -1)
        # attn_output now has the shape: (batch_size, num_targets * d_model)

        output = th.sigmoid(self.output.forward(attn_output))
        # output has the shape: (batch_size, num_targets)
        
        if not output_spans:
            return output
        
        # Get text symbol spans from the weights of words
        batch_weights: np.ndarray = attn_weights.detach().numpy()
        batch_weights = np.abs(batch_weights).max(axis=1)
        batch_weights = batch_weights - batch_weights.min()
        batch_weights = batch_weights / batch_weights.max()
        # batch_weights has now shape: (batch_size, sequence_length)
        spans = self.weights2spans(texts, batch_offsets, batch_weights, threshold)

        return output, spans
        
    def texts2batch(self, texts: List[str]) -> Tuple[th.Tensor, th.Tensor, List[List[int]]]:
        # Convert texts to batch of embeddings and padding masks
        batch_sequences = []
        batch_offsets = []
        batch_lengths = []
        max_len = 0
        for text in texts:
            tags, offsets = self.glove.text2tags(text, return_offsets=True)
            sequence = [self.glove.tag2embedding(tag) for tag in tags]
            sequence = th.tensor(sequence, dtype=th.float32, device=self.device)
            batch_sequences.append(sequence)
            batch_offsets.append(offsets)
            batch_lengths.append(len(tags))
            if max_len < len(tags):
                max_len = len(tags)
        x = pad_sequence(batch_sequences)
        # Create padding mask to zero out padded values for each sequence
        padding_mask = []
        for length in batch_lengths:
            mask = th.ones(max_len, dtype=th.float32, device=self.device)
            mask[length:] = 0.0
            padding_mask.append(mask)
        padding_mask = th.stack(padding_mask)
        return x, padding_mask, batch_offsets
    
    @staticmethod
    def weights2spans(texts: List[str],
                      batch_offsets: List[List[int]],
                      batch_weights: np.ndarray,
                      threshold=0.75
                      ) -> List[List[List[int]]]:
        # Get input words weight
        batch_spans = []
        for text, offsets, weights in zip(texts, batch_offsets, batch_weights):
            spans = []
            st, en = None, None
            for i, w in enumerate(weights):
                if i >= len(offsets):
                    break
                if (en is not None) and (en == offsets[i]):
                    continue
                if w < threshold:
                    if st is not None:
                        spans.append([st, offsets[i] - 1])
                        st, en = None, None
                    continue
                if st is None:
                    st = offsets[i]
                en = offsets[i]
            if st is not None:
                spans.append([st, len(text) - 1])
            batch_spans.append(spans)
        return batch_spans
    
    @th.no_grad()
    def predict(self, text: str, output_spans=True, threshold=0.75) -> Tuple[np.ndarray, List[List[int]]]:
        if output_spans:
            output, spans = self.forward([text], output_spans=output_spans, threshold=threshold)
            return output.detach().squeeze(0).numpy(), spans[0]
        output = self.forward([text], output_spans=False)
        return output.detach().squeeze(0).numpy()


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    This scheduler starts from zero, reaches maximum after `warmup` iterations,
    then slowly decays for `max_iters` iterations.
    It helps to 'ignite' the Transformer learning process.
    """
    def __init__(self, optimizer: optim.Optimizer, warmup: int, max_iters: int):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
    def __repr__(self):
        return f'{type(self).__name__}(warmup={self.warmup}, max_iters={self.max_num_iters})'


def perform_repair_parameters(param_groups):
    """Check and replace zero, NaN or inf parameters with random values"""
    for group in param_groups:
        for param in group['params']:
            if isinstance(param, th.Tensor):
                index = ((param.data != param.data) + (param.data == 0) +
                         (param.data == np.inf) + (param.data == -np.inf))
                n = index.sum()
                if n > 0:
                    param.data[index] = np.random.randn() / param.nelement()
                index = ((param.data < -1e+10) + (param.data > 1e+10))
                n = index.sum()
                if n > 0:
                    param.data.clamp_(min=-1e+10, max=1e+10)


class TorchWrapper(data_utils.Dataset):
    """Convert AgroCode train DataFrame into PyTorch compatible dataset"""
    def __init__(self, ds: pd.DataFrame, device: Optional[Union[str, th.device]] = None):
        self.device = device
        self.X = ds[ds.columns[1]]
        self.Y = ds[ds.columns[2:]]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X.iloc[index]
        y = self.Y.iloc[index]
        return x, th.tensor(y.to_numpy(), device=self.device, dtype=th.float32)


def log_loss_score(prediction, ground_truth):
    log_loss_ = 0
    ground_truth = np.array(ground_truth)
    for i in range(10):
        log_loss_ += log_loss(ground_truth[:, i], prediction[:, i])
    return log_loss_ / 10


def evaluate(model: nn.Module, dataset, batch_size=20) -> float:
    model.eval()
    
    index = np.arange(len(dataset))
    prediction = []
    ground_truth = []

    i = 0
    while i < len(dataset):
        # Get test batch
        e = min(i + batch_size, len(dataset))
        x, y = dataset[index[i:e]]
        y_hat = model.forward(x)
        prediction.append(y_hat.detach().numpy())
        ground_truth.append(y.numpy())
        i += batch_size
    
    prediction = np.concatenate(prediction, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    return 1 - log_loss_score(prediction=prediction, ground_truth=ground_truth)


def train(model: nn.Module,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          model_file_name: str,
          ds_train,
          ds_valid,
          batch_size=20,
          max_epochs=50,
          repair_parameters=True,
          early_stopping=True,
          tolerance=1e-5,
          patience=20,
          rng: Optional[np.random.Generator] = None,
          log: Optional[logging.Logger] = None
          ):
    if rng is None:
        rng = np.random.default_rng()
    
    if log is None:
        log = logging.getLogger()
    
    # Get number of train and test samples, batch_size
    n_train_samples, n_test_samples = len(ds_train), len(ds_valid)
    train_indexes, test_indexes = np.arange(n_train_samples), np.arange(n_test_samples)
    batch_size = int(np.clip(batch_size, 1, min(n_train_samples, n_test_samples)))
    n_train_batches = int(np.ceil(n_train_samples / batch_size))
    n_test_batches = int(np.ceil(n_test_samples / batch_size))

    
    # To keep best parameters
    best_test_loss: Optional[float] = None
    best_parameters: Optional[Dict] = None
    no_improvement_count = 0
    n_epoch = max_epochs
    n_iter = 0

    scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=25, max_iters=max_epochs * n_train_batches)
    log.info(f'Initialized scheduler:\n{repr(scheduler)}')
    
    try:
        # Iterate over epochs
        while n_epoch < max_epochs:
            # Shuffle train indexes if needed
            train_indexes = rng.permutation(train_indexes)
            
            # Reset train and test epoch indexes and registers
            train_index, test_index = 0, 0
            train_batch, test_batch = 0, 0
            accumulated_train_loss, accumulated_test_loss = 0, 0
            
            # Force clear unused memory
            gc.collect()
            
            # Iterate over batches in train and test datasets
            with tqdm.tqdm(total=(n_train_batches + n_test_batches), ncols=80) as pbar:
                while (train_index < n_train_samples) or (test_index < n_test_samples):
                    # Choose training or testing on this iteration
                    if (test_index / n_test_samples) < (train_index / n_train_samples):
                        # Perform testing:
                        
                        model.eval()
                        criterion.eval()
                        
                        # Get test batch
                        x, y = ds_valid[test_indexes[test_index:test_index + batch_size]]
                        
                        # Predict
                        y_hat = model.forward(x)
                        # Calculate overall test loss
                        loss = criterion(y_hat, y)
                        loss_scalar = loss.detach().item()
                        accumulated_test_loss += loss_scalar
                        
                        # Increment test iteration counter
                        test_index = test_index + len(x)
                        test_batch = int(np.ceil(min(n_test_samples, (test_index - 1)) / batch_size))
                    
                    else:
                        # Perform training:
                        model.train()
                        criterion.train()
                        
                        # Get next batch inputs x and targets y
                        x, y = ds_train[train_indexes[train_index:train_index + batch_size]]
                        
                        # Pass x through model and get predictions y_hat
                        y_hat = model.forward(x)
                        
                        # Calculate overall train loss
                        loss = criterion(y_hat, y)
                        loss_scalar = loss.detach().item()
                        accumulated_train_loss += loss_scalar
                        
                        # Update network weights
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Check and fix broken parameters if any
                        if repair_parameters:
                            perform_repair_parameters(optimizer.param_groups)
                        
                        # Update learning_rate if needed
                        if scheduler is not None:
                            scheduler.step()
                        
                        # Increment train iteration counter
                        train_index = train_index + len(x)
                        train_batch = int(np.ceil(min(n_train_samples, (train_index - 1)) / batch_size))
                        n_iter += 1
                    
                    pbar.update(train_batch + test_batch - pbar.n)
            
            # Compute mean train and test loss for epoch
            train_loss = accumulated_train_loss * batch_size / n_train_samples
            test_loss = accumulated_test_loss * batch_size / n_test_samples
            
            # Compute performance
            performance = evaluate(model, ds_valid, batch_size)
            
            # Increment epoch counter
            n_epoch += 1
            
            # Print epoch results
            log.info(
                f'Epoch: {n_epoch}/{max_epochs}, '
                f'iter: {n_iter}, '
                f'lr: {scheduler.get_lr()[0]:g}, '
                f'train: {train_loss:g}, '
                f'test: {test_loss:g}, '
                f'perf: {performance:g}'
            )
            
            # Check for new best result
            if (best_test_loss is None) or (best_test_loss > test_loss + tolerance):
                # Save current best parameters
                best_parameters = copy.deepcopy(model.state_dict())
                log.info(f'Saving model to {model_file_name}')
                th.save(best_parameters, model_file_name)
                no_improvement_count = 0
                best_test_loss = test_loss
            else:
                no_improvement_count += 1
            
            # Check for early stopping
            if early_stopping and (no_improvement_count > patience):
                log.info(
                    f'Test score did not improve more than tolerance={tolerance} '
                    f'for {patience} consecutive epochs. Stopping.'
                )
                break
        
        log.info('Finished training')
    
    except StopIteration:
        log.info('Training was stopped.')
    
    except KeyboardInterrupt:
        log.info('Training was interrupted by user.')
    
    except InterruptedError:
        log.info('Training was interrupted by system.')


def main():
    glove = GloveModel(substitutions=SUBSTITUTIONS, log=log)
  
    log.info('Loading train and test datasets...')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    log.info(f'Loaded train: {len(train)} records, and test: {len(test)} records')
    
    # Initialize PyTorch
    th.set_num_threads(2)
    th.set_default_dtype(th.float32)
    
    # Initialize manual random seed for reproducibility
    th.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Initialize model
    target_diseases = ('кокцидии', 'абсцесс', 'диспепсия', 'гельминтоз', 'мастит', 'ринотрахеит', 'отёк вымени',
                       'воспаление сухожилия', 'острая инфекция', 'лишай', 'заболевание')

    model = DoctorText(
        glove=glove,
        d_model=300,
        initial_targets=target_diseases,
        num_heads=4,
        num_layers=10,
        d_hidden=1024,
        max_len=5000,
        dropout=0.1,
        causal_mask=False
    )
    log.info(f'Initialized model:\n{repr(model)}')

    # Try to load weights if any
    model_file_name = 'model.pt'
    if os.path.exists(model_file_name):
        model.load_state_dict(th.load(model_file_name), strict=True)
        log.info(f'Loaded model state from {model_file_name}')
    
    # Split `train` dataset into train and validation
    ds_train, ds_valid = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED)
    log.info(f'Split dataset into train: {len(ds_train)} and valid: {len(ds_valid)}')
    
    # Measure target weights
    y_train = ds_train[ds_train.columns[2:]]
    target_weights = ((y_train.sum().sum() - y_train.sum()) / y_train.sum().sum())
    log.info(f'Target weights:\n{target_weights}')
    
    criterion = nn.BCELoss(reduction='mean', weight=th.from_numpy(target_weights.to_numpy()))
    log.info(f'Initialized criterion:\n{repr(criterion)}')
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    log.info(f'Initialized optimizer:\n{repr(optimizer)}')
    
    # Setup parameters
    params = dict(
        batch_size=20,
        max_epochs=50,
        repair_parameters=True,
        early_stopping=True,
        tolerance=1e-5,
        patience=20
    )

    # Make torch wrappers for datasets
    ds_train, ds_valid = TorchWrapper(ds_train), TorchWrapper(ds_valid)

    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          model_file_name=model_file_name,
          ds_train=ds_train,
          ds_valid=ds_valid,
          rng=rng,
          log=log,
          **params)
    
    # Construct result
    log.info('Making prediction for test dataset')
    result = {}
    for row in test.itertuples():
        output, span = model.predict(text=row.text, output_spans=True, threshold=0.75)
        result[str(row.text_id)] = {
            'label': output.tolist(),
            'span': span
        }
    print(json.dumps(result, indent=4))
    
    submission_file_name = 'submission.json'
    log.info(f'Exporting result to {submission_file_name}')
    with open(submission_file_name, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()
