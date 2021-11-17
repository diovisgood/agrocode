import os
# Reduce CPU load. Need to perform BEFORE import numpy and some other libraries.
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import json
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
from collections import OrderedDict
import requests
from tqdm import tqdm
import re

from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, cityblock, canberra, euclidean, minkowski, braycurtis
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


def log_loss_score(prediction, ground_truth):
    log_loss_ = 0
    ground_truth = np.array(ground_truth)
    for i in range(10):
        log_loss_ += log_loss(ground_truth[:, i], prediction[:, i])
    return log_loss_ / 10


def main():
    glove = GloveModel(substitutions=SUBSTITUTIONS, log=log)
  
    log.info('Loading train and test datasets...')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    log.info(f'Loaded train: {len(train)} records, and test: {len(test)} records')
    
    """
    # Обработка текстовых данных
    
    Преобразуем текст (произвольной длины) в набор фич конечной длины.
    
    Для этого придумаем некоторые ключевые слова или фразы, например: симптомы болезней.
    Найдём для них соответствующие эмбединги.
    Эмбединги каждой фразы объединим в один эмбединг путём усреднения векторов (можно и суммой, но не суть).
    
    Эти ключевые слова, эти симптомы - будут своебразными "маяками", или, если хотите, точками отсчёта.
    Каждое текстовое описание *неопределенной длины* мы заменим *конечным* набором расстояний до этих ключевых маяков.
    
    1. При анализе каждого текста мы разобьём его на **токены** - слова и знаки препинания.
    2. Далее для каждого токена найдём **эмбединг**. Для отсутствующих слов - 'unk', для чисел - 'num'.
    3. Затем вычислим **расстояние** от этого эмбединга до всех ключевых эмбедингов.
       Евклидово расстояние неинформативно в многомерном пространстве.
       Но расстояний много разных бывает. Мы будет вычислять 4 расстояния:
       ``(cosine, cityblock, euclidean, braycurtis)``
    4. При этом, для всего текста будем запоминать только **наименьшие** расстояния до каждого ключевого слова.
    5. Таким образом, из текста неограниченнной длины мы получим лишь **фиксированный набор**
       минимальных расстояний до ключевых слов.
    """

    # Symptoms keywords (or phrases)
    anchors = [
        'кокцидии', 'абсцесс', 'диспепсия', 'гельминтоз', 'мастит', 'ринотрахеит', 'отёк вымени',
        'воспаление сухожилия', 'острая инфекция', 'лишай',
        'вымя', 'сосок', 'доить', 'температура', 'шишка', 'понос', 'запор', 'кал с кровью',
        'краснота', 'слабость', 'вонь', 'буйный', 'не кушает', 'не даёт молоко', 'пьет мочу',
        'не ходит', 'не встает', 'хромает', 'орёт', 'кашляет', 'чихает', 'глаза слезятся',
        'идет пена', 'пахнет аммиаком', 'после отёла', 'вялость', 'аборт', 'свищ', 'гной из раны',
        'кровавая моча', 'выделения из носа', 'рвота', 'истощение', 'судороги', 'расширенные зрачки'
    ]
    anchor_embeddings = [np.mean(np.stack(glove[target]), axis=0) for target in anchors]

    distance_functions = (cosine, cityblock, euclidean, braycurtis)

    def embedings2features(text_embeddings: List[np.ndarray]) -> pd.Series:
        result = OrderedDict()
        for embedding in text_embeddings:
            for anchor_embedding, anchor in zip(anchor_embeddings, anchors):
                anchor = '_'.join(anchor.split())
                for dist_func in distance_functions:
                    feature_name = f'{anchor}_{dist_func.__name__}'
                    dist = float(dist_func(embedding, anchor_embedding))
                    if feature_name not in result:
                        result[feature_name] = dist
                    else:
                        result[feature_name] = min(dist, result[feature_name])
        return pd.Series(result)

    def embedings2distances(text_embeddings: List[np.ndarray]) -> Tuple[pd.Series, np.ndarray]:
        result = OrderedDict()
        distances = []
        for embedding in text_embeddings:
            D = np.ones((len(anchor_embeddings), len(distance_functions)), dtype=np.float32) * np.inf
            for i, (anchor_embedding, anchor) in enumerate(zip(anchor_embeddings, anchors)):
                anchor = '_'.join(anchor.split())
                for j, dist_func in enumerate(distance_functions):
                    feature_name = f'{anchor}_{dist_func.__name__}'
                    dist = float(dist_func(embedding, anchor_embedding))
                    D[i, j] = dist
                    if feature_name not in result:
                        result[feature_name] = dist
                    else:
                        result[feature_name] = min(dist, result[feature_name])
            distances.append(D)
        return pd.Series(result), np.stack(distances, axis=0)

    def text2features(text) -> pd.Series:
        text_embeddings = glove[text]
        return embedings2features(text_embeddings)

    def init_dataset(ds: pd.DataFrame, file_name: str) -> pd.DataFrame:
        # Check if file already exists
        if not os.path.exists(file_name):
            log.info(f'Constructing new synthetic dataset...')
            tqdm.pandas(position=0, leave=True)
            X: pd.DataFrame = ds['text'].progress_apply(text2features)
        
            data = [ds[['text_id']], X]
            if len(ds.columns) > 2:
                data.append(ds[ds.columns[2:]])
        
            data = pd.concat(data, axis=1, ignore_index=False, copy=False)
        
            data.to_feather(file_name)
            log.info(f'Saved new dataset to {file_name}')
    
        else:
            data = pd.read_feather(file_name)
            log.info(f'Loaded synthetic dataset from {file_name}')
    
        assert len(data) == len(ds)
        return data

    log.info(f'Constructing new features based on text embeddings...')
    data = init_dataset(train, 'train.synth.feather')
    log.info(f'New synthetic {len(data)} features:\n{data.columns}')

    # Split `train` dataset into train and validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        data[data.columns[1:-11]],
        data[data.columns[-11:]],
        test_size=0.2,
        random_state=RANDOM_SEED
    )
    X_train, y_train = data[data.columns[1:-11]], data[data.columns[-11:]]
    log.info(f'Split dataset into train: {len(X_train)} and valid: {len(X_valid)}')
    
    # Initialize and train new model
    estimator = CatBoostClassifier(
        max_depth=5,
        iterations=100,
        # bootstrap_type=,  # Bayesian, Bernoulli, MVS,
        # boosting_type='Ordered',
        # loss_function='MultiLogloss',
        # eval_metric='MultiLogloss',
        logging_level='Verbose',
        thread_count=2,
        allow_writing_files=False,
        random_seed=RANDOM_SEED,
    )

    model = OneVsRestClassifier(estimator=estimator)
    log.info(f'Initialized model: {model}')

    log.info(f'Starting training...')
    model.fit(X_train, y_train)
    log.info('Done')
    
    # Get model score on validation set
    y_pred = model.predict_proba(X_valid)
    score = log_loss_score(ground_truth=y_valid, prediction=y_pred)
    log.info(f'Model score on validation set: {score}')
    
    # Compute features importance
    target_features_importance = [estimator.feature_importances_ for estimator in model.estimators_]
    target_features_importance = abs(np.stack(target_features_importance, axis=0))
    target_features_importance = target_features_importance / target_features_importance.sum(0, keepdims=True)
    target_features_importance = np.nan_to_num(target_features_importance, nan=0.0, posinf=1.0, neginf=0.0)
    # target_features_importance has shape: (num_targets, num_features)
    # for each estimator, in each row, the sum of coefficients is equal to 1.0

    # Construct result
    log.info('Constructing result...')
    result = {}
    target_threshold = 0.6
    token_threshold = 0.65

    with tqdm(total=len(test), position=0, leave=True) as pbar:
        for row in test.itertuples():
            tags, offsets = glove.text2tags(row.text, return_offsets=True)
            text_embeddings = [glove.tag2embedding(tag) for tag in tags]
            input, distances = embedings2distances(text_embeddings)
            # distances has shape: (num_tokens, num_symptoms, num_distance_functions)

            distances = abs(distances.reshape((distances.shape[0], -1)))
            # distances has shape: (num_tokens, num_features)
            
            output = model.predict_proba([input])[0]
            # output has shape: (num_targets,)

            # Zero out some targets below threshold
            targets_importance = output.copy()
            targets_importance[targets_importance < target_threshold] = 0.0
            # targets_importance has shape: (num_targets,)

            features_importance = np.expand_dims(targets_importance, 1) * target_features_importance
            # features_importance has shape: (num_targets, num_features)

            # Invert distances, as we are interested in the closest distance.
            # Note: here I assume that the shorter the distance - the more important the token is.
            # But, in fact, we don't know at what distance the feature becomes important!
            inverted_distances = distances.max(0, keepdims=True) - distances
            inverted_distances = (inverted_distances - inverted_distances.min(0, keepdims=True)) / inverted_distances.max(0, keepdims=True)
            # inverted_distances has shape: (num_tokens, num_features)

            token_importance = inverted_distances @ features_importance.transpose()
            # token_importance has shape: (num_tokens, num_targets)
            
            token_importance = token_importance.sum(1)
            token_importance = token_importance / token_importance.max()
            token_importance = np.nan_to_num(token_importance, nan=0.0, posinf=1.0, neginf=0.0)
            # token_importance has shape: (num_tokens,)
            
            span = []
            st, en = None, None
            for i, (importance, offset) in enumerate(zip(token_importance, offsets)):
                # Some consequent tokens address the same starting offset in the text.
                # This is due to words substitutions by phrases for missing words in Glove.
                if (en is not None) and (en == offset):
                    continue
                # Current word is not important
                if importance < token_threshold:
                    if st is not None:
                        en = offset - 1
                        if en - st > 1:
                            span.append([st, en])
                        st, en = None, None
                    continue
                # Current word is the first important word in the span. Keep st
                if st is None:
                    st = offset
                # Update en
                en = offset
            if st is not None:
                en = len(row.text) - 1
                if en - st > 1:
                    span.append([st, len(row.text) - 1])
                    
            result[str(row.text_id)] = {
                'span': span,
                'label': output[:10].tolist(),
            }
            pbar.update(1)
    log.info('Done')

    submission_file_name = 'submission.json'
    log.info(f'Exporting result to {submission_file_name}')
    with open(submission_file_name, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()
