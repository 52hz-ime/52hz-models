"""Segment raw pinyin keys into syllables, tolerating one mistyped letter per syllable.

The result is a hint for the model, not a reading: the keys stay the only source of intent.
"""

import re


SYLLABLES = frozenset("""a ai an ang ao ba bai ban bang bao bei ben beng bi bian biang biao bie bin bing bo bu ca cai can cang cao ce
cei cen ceng cha chai chan chang chao che chen cheng chi chong chou chu chua chuai chuan chuang chui chun chuo ci cong cou cu
cuan cui cun cuo da dai dan dang dao de dei den deng di dia dian diao die din ding diu dong dou du duan dui dun duo e ei en eng
er fa fan fang fei fen feng fiao fo fou fu ga gai gan gang gao ge gei gen geng gong gou gu gua guai guan guang gui gun guo ha hai
han hang hao he hei hen heng hm hng hong hou hu hua huai huan huang hui hun huo ji jia jian jiang jiao jie jin jing jiong jiu ju
juan jue jun ka kai kan kang kao ke kei ken keng kong kou ku kua kuai kuan kuang kui kun kuo la lai lan lang lao le lei len leng
li lia lian liang liao lie lin ling liu lo long lou lu luan lun luo lv lve m ma mai man mang mao me mei men meng mi mian miao mie
min ming miu mo mou mu n na nai nan nang nao ne nei nen neng ng ni nia nian niang niao nie nin ning niu nong nou nu nuan nun nuo
nv nve o ou pa pai pan pang pao pei pen peng pi pian piao pie pin ping po pou pu qi qia qian qiang qiao qie qin qing qiong qiu qu
quan que qun ran rang rao re ren reng ri rong rou ru rua ruan rui run ruo sa sai san sang sao se sen seng sha shai shan shang
shao she shei shen sheng shi shou shu shua shuai shuan shuang shui shun shuo si song sou su suan sui sun suo ta tai tan tang tao
te tei teng ti tian tiao tie ting tong tou tu tuan tui tun tuo wa wai wan wang wei wen weng wo wong wu xi xia xian xiang xiao xie
xin xing xiong xiu xu xuan xue xun ya yan yang yao ye yi yin ying yo yong you yu yuan yue yun za zai zan zang zao ze zei zen
zeng zha zhai zhan zhang zhao zhe zhei zhen zheng zhi zhong zhou zhu zhua zhuai zhuan zhuang zhui zhun zhuo zi zong zou zu zuan
zui zun zuo""".split())
# How surprising each syllable is, in hundredths of a nat, counted over about 55,000 syllables of benchmark characters;
# common readings such as de or shi are preferred when two cuttings of the keys are otherwise equally good.
SURPRISE_TEXT = """a:543 ai:634 an:677 ang:1001 ao:733 ba:559 bai:666 ban:583 bang:695 bao:597 bei:558 ben:625 beng:922 bi:574
bian:583 biang:1162 biao:652 bie:660 bin:867 bing:652 bo:668 bu:397 ca:1162 cai:585 can:713 cang:867 cao:891 ce:736
cei:1162 cen:1162 ceng:769 cha:702 chai:942 chan:695 chang:553 chao:667 che:682 chen:733 cheng:548 chi:570 chong:700 chou:754
chu:532 chua:1162 chuai:1052 chuan:668 chuang:741 chui:848 chun:786 chuo:1162 ci:592 cong:693 cou:967 cu:891 cuan:1052 cui:878
cun:695 cuo:715 da:491 dai:591 dan:574 dang:659 dao:458 de:319 dei:1162 den:1162 deng:636 di:518 dia:1162 dian:498
diao:773 die:812 din:1162 ding:624 diu:942 dong:568 dou:509 du:608 duan:682 dui:552 dun:840 duo:534 e:777 ei:1052
en:825 eng:1162 er:539 fa:557 fan:612 fang:559 fei:630 fen:589 feng:656 fiao:1162 fo:967 fou:848 fu:535 ga:867
gai:614 gan:588 gang:681 gao:600 ge:439 gei:617 gen:681 geng:671 gong:553 gou:677 gu:624 gua:806 guai:757 guan:584
guang:687 gui:679 gun:922 guo:471 ha:618 hai:485 han:695 hang:718 hao:488 he:520 hei:754 hen:547 heng:867 hm:1162
hng:1162 hong:730 hou:534 hu:636 hua:551 huai:790 huan:599 huang:751 hui:505 hun:773 huo:580 ji:439 jia:504 jian:517
jiang:580 jiao:562 jie:523 jin:516 jing:518 jiong:1162 jiu:482 ju:563 juan:891 jue:585 jun:711 ka:733 kai:602 kan:514
kang:801 kao:704 ke:509 kei:1162 ken:832 keng:891 kong:696 kou:738 ku:711 kua:840 kuai:674 kuan:765 kuang:801 kui:806
kun:848 kuo:848 la:652 lai:517 lan:700 lang:777 lao:639 le:410 lei:686 len:1162 leng:812 li:474 lia:781 lian:631
liang:585 liao:652 lie:801 lin:695 ling:673 liu:640 lo:1162 long:747 lou:795 lu:638 luan:867 lun:715 luo:689 lv:682
lve:867 m:1162 ma:522 mai:644 man:684 mang:786 mao:711 me:507 mei:485 men:540 meng:725 mi:664 mian:587 miao:790
mie:1001 min:733 ming:565 miu:1162 mo:641 mou:777 mu:655 n:867 na:500 nai:781 nan:600 nang:1162 nao:801 ne:602
nei:646 nen:1052 neng:559 ng:1162 ni:435 nia:1162 nian:550 niang:922 niao:942 nie:1001 nin:825 ning:867 niu:747 nong:751
nou:1162 nu:825 nuan:867 nun:1162 nuo:840 nv:666 nve:967 o:706 ou:812 pa:747 pai:663 pan:744 pang:781 pao:736
pei:706 pen:922 peng:711 pi:736 pian:618 piao:689 pie:1162 pin:642 ping:661 po:736 pou:1052 pu:769 qi:498 qia:967
qian:566 qiang:730 qiao:801 qie:700 qin:704 qing:571 qiong:1001 qiu:632 qu:507 quan:638 que:686 qun:832 ran:625 rang:689
rao:878 re:722 ren:473 reng:832 ri:651 rong:700 rou:781 ru:611 rua:1162 ruan:891 rui:891 run:891 ruo:781 sa:812
sai:751 san:660 sang:891 sao:905 se:711 sen:857 seng:1001 sha:676 shai:942 shan:671 shang:509 shao:632 she:659 shei:922
shen:531 sheng:537 shi:321 shou:535 shu:564 shua:801 shuai:786 shuan:1162 shuang:757 shui:608 shun:840 shuo:544 si:551 song:722
sou:832 su:687 suan:711 sui:657 sun:806 suo:645 ta:480 tai:573 tan:741 tang:725 tao:704 te:687 tei:1162 teng:795
ti:562 tian:546 tiao:695 tie:769 ting:599 tong:587 tou:634 tu:660 tuan:786 tui:715 tun:1001 tuo:761 wa:757 wai:709
wan:551 wang:594 wei:476 wen:587 weng:1052 wo:405 wong:1162 wu:543 xi:499 xia:558 xian:514 xiang:486 xiao:509 xie:581
xin:545 xing:535 xiong:769 xiu:700 xu:613 xuan:695 xue:599 xun:777 ya:612 yan:549 yang:566 yao:509 ye:492 yi:368
yin:568 ying:530 yo:922 yong:606 you:407 yu:510 yuan:522 yue:576 yun:695 za:790 zai:470 zan:757 zang:891 zao:684
ze:711 zei:1001 zen:668 zeng:754 zha:840 zhai:801 zhan:637 zhang:580 zhao:626 zhe:442 zhei:1162 zhen:597 zheng:583 zhi:428
zhong:509 zhou:641 zhu:553 zhua:857 zhuai:1162 zhuan:646 zhuang:691 zhui:848 zhun:757 zhuo:840 zi:494 zong:709 zou:698 zu:679
zuan:1052 zui:569 zun:942 zuo:536"""
SURPRISE = {item.split(":")[0]: int(item.split(":")[1]) for item in SURPRISE_TEXT.split()}
PRIOR = 0.15
ROWS = (("qwertyuiop", 0.0), ("asdfghjkl", 0.25), ("zxcvbnm", 0.75))
COORDINATES = {char: (index + offset, row) for row, (keys, offset) in enumerate(ROWS) for index, char in enumerate(keys)}
NEIGHBORS = {char: sorted(other for other, point in COORDINATES.items() if other != char and (
    (COORDINATES[char][0] - point[0]) ** 2 + (COORDINATES[char][1] - point[1]) ** 2) ** 0.5 <= 1.2) for char in COORDINATES}
SHORTENED = {}
for syllable in SYLLABLES:
    for index in range(len(syllable)):
        shortened = syllable[:index] + syllable[index + 1:]
        if shortened not in SYLLABLES:
            SHORTENED.setdefault(shortened, []).append(syllable)
# The same index without that exception: a valid syllable may still be a longer one with a letter dropped.
DROPPED = {}
for syllable in SYLLABLES:
    for index in range(len(syllable)):
        shortened = syllable[:index] + syllable[index + 1:]
        if len(shortened) > 1:
            DROPPED.setdefault(shortened, []).append(syllable)
LONGEST = max(map(len, SYLLABLES))
VOWELS = frozenset("aeiouv")
INITIALS = frozenset("b p m f d t n l g k h j q x r z c s y w zh ch sh".split())
# Each tolerated error has a prior cost; a syllable that lost its vowel is less likely than one that lost another letter.
EXACT, SWAPPED, DUPLICATE, MISSING, EXTRA, NEIGHBOR, SWAPPED_PAIR, MISSING_VOWEL, STRAY_VOWEL, STRAY_CONSONANT = (
    0.0, 0.8, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 2.6)
# A valid syllable may still stand for a neighbouring one when the typist merges two sounds.
ACCENT = 0.6
RARE = 1.5
UNRELIABLE = 0.4
# For alternatives at one spot: readings within this of the cheapest one are kept.
MARGIN = 1.2
RARE_SYLLABLES = frozenset(("m", "n", "ng", "hm", "hng"))
RUNS = re.compile(r"[A-Z]+[a-z]*|[a-z]+|[0-9]+|[^A-Za-z0-9]")


def commonest(candidates):
    """Pick the least surprising syllable among equally valid readings."""
    return min(candidates, key=lambda item: SURPRISE.get(item, 1500))


def edits(run, start, size):
    """Return the cost of reading run[start:start + size] as one syllable, and the syllable it points to."""
    span = run[start:start + size]
    if span in SYLLABLES:
        return (RARE if span in ("m", "n", "ng", "hm", "hng") else EXACT), span
    if size == 1:
        if (start > 0 and run[start - 1] == span) or (start + 1 < len(run) and run[start + 1] == span):
            return DUPLICATE, span
        return (STRAY_VOWEL if span in VOWELS else STRAY_CONSONANT), span
    swapped = [span[:index] + span[index + 1] + span[index] + span[index + 2:] for index in range(size - 1)]
    if any(item in SYLLABLES for item in swapped):
        return SWAPPED if size > 2 else SWAPPED_PAIR, commonest(item for item in swapped if item in SYLLABLES)
    if span in SHORTENED and VOWELS.intersection(span):
        return MISSING, commonest(SHORTENED[span])
    if span in INITIALS:
        return MISSING_VOWEL, span
    extra = [span[:index] + span[index + 1:] for index in range(size)]
    if any(item in SYLLABLES for item in extra):
        return EXTRA, commonest(item for item in extra if item in SYLLABLES)
    for index, char in enumerate(span):
        replaced = [span[:index] + other + span[index + 1:] for other in NEIGHBORS.get(char, ())]
        if any(item in SYLLABLES for item in replaced):
            return NEIGHBOR, commonest(item for item in replaced if item in SYLLABLES)
    return None


def accent_partners(syllable):
    """What a valid syllable may stand for when the typist does not separate two sounds."""
    if len(syllable) < 2:
        return []
    partners = []
    if syllable.endswith("ng"):
        partners.append(syllable[:-1])
    elif syllable + "g" in SYLLABLES:
        partners.append(syllable + "g")
    if syllable[0] in "nl" and syllable != "ng":
        partners.append(("l" if syllable[0] == "n" else "n") + syllable[1:])
    plain = {"z": "zh", "c": "ch", "s": "sh"}.get(syllable[0], "")
    if plain:
        partners.append(plain + syllable[1:])
    if syllable[0] in "fh":
        partners.append(("h" if syllable[0] == "f" else "f") + syllable[1:])
    return [partner for partner in partners if partner in SYLLABLES and partner != syllable]


def readings(run, start, size, accent=True):
    """Every syllable the keys at one span can stand for, cheapest first, within the margin of the cheapest."""
    span = run[start:start + size]
    found = {}

    def add(cost, syllable):
        if syllable not in found or cost < found[syllable]:
            found[syllable] = cost

    if span in SYLLABLES:
        add(RARE if span in RARE_SYLLABLES else EXACT, span)
        if accent:
            for partner in accent_partners(span):
                add(ACCENT, partner)
    if size == 1:
        if (start > 0 and run[start - 1] == span) or (start + 1 < len(run) and run[start + 1] == span):
            add(DUPLICATE, span)
        else:
            add(STRAY_VOWEL if span in VOWELS else STRAY_CONSONANT, span)
        return sorted((cost, syllable) for syllable, cost in found.items())
    for index in range(size - 1):
        swapped = span[:index] + span[index + 1] + span[index] + span[index + 2:]
        if swapped in SYLLABLES:
            add(SWAPPED if size > 2 else SWAPPED_PAIR, swapped)
    if span in DROPPED and VOWELS.intersection(span):
        for full in DROPPED[span]:
            add(MISSING, full)
    if span in INITIALS:
        add(MISSING_VOWEL, span)
    for index in range(size):
        shortened = span[:index] + span[index + 1:]
        if shortened in SYLLABLES:
            add(EXTRA, shortened)
    for index, char in enumerate(span):
        for other in NEIGHBORS.get(char, ()):
            replaced = span[:index] + other + span[index + 1:]
            if replaced in SYLLABLES:
                add(NEIGHBOR, replaced)
    if not found:
        return []
    cheapest = min(found.values())
    return sorted((cost, syllable) for syllable, cost in found.items() if cost <= cheapest + MARGIN)[:8]


def alternatives(typed, exclude):
    """Other syllables the typed letters can stand for, most plausible first."""
    if not typed.isalpha() or not 1 <= len(typed) <= LONGEST:
        return []
    found = readings(typed, 0, len(typed))
    found.sort(key=lambda item: item[0] + PRIOR * SURPRISE.get(item[1], 1500) / 100)
    return [syllable for _, syllable in found if syllable not in exclude][:4]


def segment(keys):
    """Split keys into hint tokens: syllables with a '?' where a keystroke error was tolerated."""
    tokens = []
    for match in RUNS.finditer(keys):
        token = match.group()
        if token.islower():
            tokens.extend(segment_run(token))
        elif token.isalpha() and not token.isupper():
            # A capitalised run is pinyin when it reads cleanly, otherwise it is a word such as iPhone or Atlas.
            pieces = segment_run(token.lower())
            tokens.extend(pieces if len(pieces) > 1 and all(not piece.endswith("?") for piece in pieces) else [token])
        else:
            tokens.append(token)
    return " ".join(tokens)


def segment_run(run):
    """Cut one lowercase run into syllables with the fewest and cheapest tolerated errors."""
    count = len(run)
    best = [(0.0, 0)] + [None] * count
    token_at = [None] * (count + 1)
    errors = [0.0] + [None] * count
    for end in range(1, count + 1):
        for size in range(1, min(LONGEST, end) + 1):
            start = end - size
            reading = edits(run, start, size)
            if reading is None or best[start] is None:
                continue
            cost, syllable = reading
            score = (round(best[start][0] + cost + PRIOR * SURPRISE.get(syllable, 1500) / 100, 4), best[start][1] + 1)
            if best[end] is None or score < best[end]:
                best[end] = score
                token_at[end] = (start, cost, syllable)
                errors[end] = errors[start] + cost
    if errors[count] > UNRELIABLE * count:
        return [run + "?"]
    tokens, end = [], count
    while end:
        start, cost, syllable = token_at[end]
        tokens.append((syllable or run[start:end]) + ("?" if cost else ""))
        end = start
    return tokens[::-1]
