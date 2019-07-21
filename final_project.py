import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np
import math
from numpy import linalg
import datetime
import os


margins = 4
w = 1240
h = 1754
boxes = [[116, 175], [271, 341], [421, 481], [534, 590], [647, 708], [763, 821], [875, 934], [988, 1049], [1102, 1162], [1215, 1276], [1330, 1389], [1443, 1504], [1562, 1638]]
MAX_CHAN = 255
RNG_ERROR = 5
WHITE_RGB = [MAX_CHAN, MAX_CHAN, MAX_CHAN]
RNG_MAX = MAX_CHAN - RNG_ERROR
NUM_OF_PIX_RECT = 400
RNG_ERR_RECT = 0.5
EDGE_SIZE = 21
MAX_BLACK_CHAN = 100
FULL_ROUND = 180
MID_OF_IMG = 570
DIF_REG_LINES = 230
RANGE_MID = range(400, MID_OF_IMG)
PRINT_BEG = 13
PRINT_BEG_MAX = PRINT_BEG + 3
PADDING_DOWN_LINE = 30
PADD = 3
avg_letter_size = 20
rectangle_side = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

term_A = ['איך נטוס עם גד כץ, הזקן שמחלף בצרפת?', 'עטלף אבק נס דרך מזגן שהתפוצץ כי חם.', 'מנעולן הפך כף חצץ שגזר קט איבד סתם.', 'גנב שהעז אכל חינם צדף קר וטס', 'חסכן קל דעת בארץ פשוט הגזים', 'פריחת נטע שסק בצד ואגוז המלך', 'איך נטוס עם גד כץ, הזקן שמחלף בצרפת?', 'עטלף אבק נס דרך מזגן שהתפוצץ כי חם', 'מנעולן הפך כף חצץ שגזר קט איבד סתם.', 'לביא טרף גמד זקן שחסך הצעות', 'קטנטן צפצף כפכף שומן קוץ חפץ תם צץ רך']
term_B = ['איך בלש תפס גמד רוצח עז קטנה.', 'שפן בלי כף אכל קצת גזר בטעם חסה, ודי.', 'השפן טעם ביס ואכל קצת גזר חד.', 'קזחסטן ארץ מעלפת, גדושה בכי.', 'פז, שרת קמצן, בוגד החליט: אכעס עליך', 'תמר אכלה גז פוק, בעץ טניס חדש.', 'ממבל קנטרן שזעף פצץ אגד וכך הסתים.', 'החזיר רץ, מצא כספת ובלע דג קטן', 'חסכן קל דעת בארץ פשוט הגזים', 'צפע חזק נשך דג מת באוסטרליה', 'קטנטן צפצף כפכף שומן קוץ חפץ תם צץ רך']
term_C = ['נחש צפע קט בהזיות, דאג מסל רך.', 'כפיל שקט נסע בגדה, צרח ואז מת מהתקף', 'איש עם זקן טס לצרפת ודג בחכה.', 'פגז מחק את העצל שבכורדיסטן.', 'הקצין גד חזר ותבע כיסא למשפט.', 'חלזון גֹרש כי התעסק בצדף טמא.', 'עצבת-חטא סרק, יזום כנגד פשלה.', 'לביא טרף גמד זקן תמים שחסך הצעות.', 'הצדק פגש את טמבל נוסע כחזיר.', 'גנב שהעז אכל חינם צדף קר וטס.', 'קטנטן צפצף כפכף בוץ קוץ חפץ תמים לך רך']
term_D = ['איש מצטדק פגע בזונה חסרת כל.', 'זקן גס ליטף את הכרוב עד שצמח.', 'זקן, אך גס-רוח בדם, יעץ להתפשט.', 'חסכן קל דעת בארץ פשוט הגזים.', 'הזר שטיגן תפוד בכלא צעק חמס.', 'אין חזך רופס למעט קצת גב השד.', 'פריחת נטע שסק בצד אגוז המלך.', 'אני זוכרת שטל ספק בגדה עם צח.', 'גמד קנטרן שזעף פצץ אגד וכך הסתים.', 'סכנה! זאב גדול עצל קפץ דרך מטף מים חם.', 'קטנטן צפצף כפכף בוץ קוץ חפץ תמים לך רך']
term_E = ['כהרף־עין נשבט לך גוץ חוצפן. תפוס חזק!', 'הרכז צמא, ניסח: בלעת קטשופ, דג.', 'גד פושטק! תעלב? חסין אומץ זכרה.', 'אגדוש, הכן טבח צרפתי! מעז, קלס.', 'סלק זעם! יתפרץ, חבט נכה. שוד גא.', 'לכסות טיפש המצע יחזק. דג נברא.', 'ארב נגד, קזחי עָצמה שפט. תוסכל.', 'גוף סבל, זעקתם: הכחיד ארץ שטן!', 'נטש צר? אדיח. כה מתק עוז – לב ספוג.', 'מנעולן הפך כף חצץ שגזר קט איבד סתם.', 'מקומם צפוף כיף בוץ קוץ חפץ תמים לך רך']
term_F = ['בעמק יפה בין כרמים ושדות', 'עומד מגדל בן חמש קומות . ומי גר במגדל?', 'בקומה הראשונה תרנגולת שמנה,', 'כל היום בביתה על משכבה מתהפכת.', 'היא כל-כך שמנה שקשה לה ללכת.', 'בקומה השנייה גרה קוקייה, כל היום היא מהלכת.', 'עושה ביקורים ,כי בניה גרים בקינים אחרים.', 'בקומה השלישית חתולה כושית נקייה, מגונדרת. על צוואר יש לה סרט.', 'בקומה הרביעית גרה סנאית.', 'בשמחה ונחת אגוזים מפצחת.', 'ובקומה החמישית גר מר עכבר.']
term_G = ['אך לפני שבוע ארז חפציו ונסע', 'איש אינו יודע לאן ומדוע.', 'כתבו דיירי המגדל שלט, תקעו מסמר מעל לדלת', 'וקבעו שלט בקיר: דירה להשכיר', 'והנה בשבילים ,בדרכים, בכבישים.', 'אל הבית באים דיירים חדשים.', 'ראשונה באה נמלה, לקומה חמישית עולה, קורא את השלט,', 'פותחת את הדלת, עומדת בפנים ומסתכלת.', 'באים מכל הדירות השכנים, עומדים מסביב, מסבירים לה פנים', 'הנאים החדרים בעיניך?', 'אומרת הנמלה: השכנים אינם טובים בעיני.']
term_H = ['איך אשב פה, אם לעשרים ארנבות, עם קוקייה מפקירה הבנים', 'כל בניה גדלו בקינים זרים, כולם עזובים, כולם מופקרים', 'הלכה הארנבת בא החזיר,', 'אחרי שקרא את השלט, התגלגל ועלה ופתח את הדלת', 'עומד ומביט העניו הקטנות בכתלים, בתקרה ובחתונות.', 'באים מכל הדירות השכנים, עומדים סביבו, ומסבירים לו פנים', 'בא זמיר. שר הזמיר בקול רינה, עולה הזמיר לקומה אחרונה.', 'עולה קול פיצוחם עד לב השמיים, איום ונורא, מחריש אוזניים', 'ואזני רגילות לקולות אחרים, רק לשירים ולמזמורים.', 'באה יונה. חיש קל עלתה לקומה אחרונה', 'שכרה היונה את הדירה, יום- יום הומה היא בחדרה.']
term_I = ['פעם אחת , לפני הרבה שנים, חי מלך שאהב מאוד בגדים יפים', 'ארון הבגדים שלו היה כה חשוב בעיניו , שהיו לו יותר חייטים מחיילים', 'בכל יום היה עורך טכס חגיגי ,', 'רק כדי להתפאר בבגדיו המגונדרים', 'ולכל שעה משעות היום היה לו בגד מיוחד', 'האנשים בממלכה נאלצו ללכת לטקסים כה רבים', 'שלא נשאר להם זמן לעבוד', 'אבל המלך לא הבין איזה נזק הוא גורם לממלכה', 'הוא המשיך לצעוד ברחובות ולהתפאר', 'בירת הממלכה הייתה גדולה ויפה', 'תיירים באו מרחוק כדי להתפעל ממנה']
term_J = ['והמלך ניצל זאת כדי לערוך עוד ועוד טכסים ומסיבות', 'החיטים שלו נאלצו לעבוד מסביב לשעון', 'כדי לעמוד בדרישותיו למחלצות חדשות', 'חדר ההלבשה של המלך היה ענקי , כמו אולם ריקודים', 'כל בוקר התבונן המלך בשפע הגלימות', 'כל בוקר התבונן המלך בשפע הגלימות', 'משרתיו המתינו בדאגה מחוץ לדלת , עד שיחליט', 'יום אחד הופיעו בעיר שני זרים', 'שועל בשם "מחט" , וקוף שקרא לעצמו "חוט', 'הם התיימרו להיות חיטים מומחים מארץ רחוקה', 'אשר באו לארג למלך מערכת בגדים , שאין יפה ממנה בעולם כולו.']
term_K = ['למעשה , הם היו רמאים ונוכלים', 'כשהגיעו לארמון, השתחוו חוט ומחט לפני המלך ואמרו', 'הוד מלכותך , הבד שלנו לא רק מקסים ונהדר', 'יש לו גם תכונה מופלאה: טיפשים לא יוכלו לראותו.', 'מצוין! חשב המלך , שילוב של יופי וחוכמה.', 'ומיד ציווה על החיטים להתחיל במלכה.', 'לאחר שקיבלו המון כסף כדי לקנות חומרים', 'התיישבו חוט ומחט מול הנול והעמידו פנים שהם אורגים.', 'חוטי הכסף והזהב שהם הזמינו היו מוסתרים בחדרם.', 'אבל רחש הנול הריק שלהם נשמע בארמון עד מאוחר בלילה.', 'כולם התרשמו מחריצותם הרבה.']
term_L = ['המלך התפוצץ מסקרנות בקשר לבגדיו החדשים.', 'לבסוף שלח את יועצו הבכיר ביותר', 'לראות איך מתקדמת העבודה.', 'היועץ המסכן נדהם: הוא לא ראה את הבד', 'שהאורגים הציגו לפניו בגאווה.', 'האומנם אני טיפש כזה? חשב בדאגה.', 'אעמיד פנים שראיתי.', 'כדי שהמלך לא יחשוד בו , חזר היועץ ובפיו הלולים וחשבונות', 'העבודה מתקדמת להפליא , מלכי.', 'אילו דוגמאות , אילו צבעים! עין לא ראתה', 'המלך היה מאשר']
term_M = ['סופסוף הגיע היום הגדול.', 'בראש , מתחת לחופה מיוחדת , צעד המלך – עירום , פרט לכותנתו.', 'אנשי העיר הנרגשים הצטופפו בצידי הדרך.', 'אבל לפתע נשמע קול קטן וצלול מעל רחש ההמון.', 'אבא , אמר הקול בפליאה, המלך לא לובש שום דבר!', 'האב נבוך ניסה להשקיט את ילדו.', 'אל תדבר שטויות , לחש.', 'אבל הילד חזר על דבריו בכל רם.', 'והקהל הצטרף אליו: המלך עירום.', 'מרוב שמחה אין עוד צורך להעמיד פנים.', 'פרץ הקהל כולו בצחוק גדול.']

def main_func(name):

    # getting the img, size, and the data:
    data, img = get_image(name)
    # save coordinates of the black rectangles
    cords = save_cords(data)
    # perspective transform on the margins
    aligned = get_perspective(img, cords)
    # rotate 180 deg the img if needed, and load matrix in order to manipulate image
    aligned_data, aligned = check_opp(aligned)
    symbol_of_term = recognize_chars(aligned_data)
    print(symbol_of_term, name)
    term = from_char_to_term(symbol_of_term)
    written_ranges, sides, components,words,words_ind = find_lines_words_component(aligned_data, term)
    tree = xml(written_ranges, name, sides, components,words, words_ind, term)
    aligned = Image.fromarray(aligned_data, 'RGB')
    return aligned, tree


# getting by image name the properties of img
def get_image(img_name):
    img = Image.open(img_name)
    img = img.resize((w, h), Image.ANTIALIAS)
    data = np.array(img)
    return data, img


# method that manage the search of alignments exact coordinates
def save_cords(data):
    # where to look at(x,y axis)
    margins_side = math.floor(w / margins)
    margins_vertical = math.floor(h / margins)
    # up-left - no need to flip
    arr_u_l = convert_to_bin_array(data, 0, margins_vertical, 0, margins_side)
    i_u_l, j_u_l = find_edges(arr_u_l, False)
    # up-right - flip the x axis
    arr_u_r = convert_to_bin_array(data, 0, margins_vertical, w - margins_side, w)
    arr_u_r = flip_by_axis(arr_u_r, True, False)
    i_u_r, j_u_r = find_edges(arr_u_r, False)
    # back to original coordinates
    j_u_r = w - j_u_r
    # down-left - flip in y axis
    arr_d_l = convert_to_bin_array(data, h - margins_vertical, h, 0, margins_side)
    arr_d_l = flip_by_axis(arr_d_l, False, True)
    i_d_l, j_d_l = find_edges(arr_d_l, True)
    # back to original coordinates
    i_d_l = h - i_d_l
    # down_right - flip in both axises
    arr_d_r = convert_to_bin_array(data,  h - margins_vertical, h, w - margins_side, w)
    arr_d_r = flip_by_axis(arr_d_r, True, True)
    i_d_r, j_d_r = find_edges(arr_d_r, True)
    # back to original in both axises
    j_d_r = w - j_d_r
    i_d_r = h - i_d_r
    # put all the data in the array
    arr = [(j_u_l, i_u_l), (j_u_r, i_u_r), (j_d_r, i_d_r), (j_d_l, i_d_l)]
    return arr



# its more convenient to look in 2d binary array than in 3d rgb
def convert_to_bin_array(data, from_i, to_i, from_j, to_j):
    arr = []
    # just in specific range in margins
    for i in range(from_i, to_i):
        line = []
        for j in range(from_j, to_j):
            col_to_app = 0
            if is_black(data[i][j]):
                col_to_app = 1
            line.append(col_to_app)
        arr.append(line)
    return arr


# check by definition of black color(lowest channel), can be updated by need
def is_black(pix):
    for ind in range(3):
        if pix[ind] > MAX_BLACK_CHAN:
            return False
    return True


# flip binaric array in order to look from [0,0] in the bin array in each margin
def flip_by_axis(arr, right, down):
    if right:
        arr = np.flip(arr, 1)
    if down:
        arr = np.flip(arr, 0)
    return arr


# find the first sharp "change" - the edge
def find_edges(arr, b):
    range_i, range_j, range_delta = len(arr), len(arr[0]), 1
    rng_matrix = EDGE_SIZE - range_delta
    if b:
        tmp = range_i
        range_i, range_j = range_j, tmp
    for i in range(range_delta, range_i-rng_matrix):
        for j in range(range_delta, range_j-rng_matrix):
            if edge_matrix(arr, i, j, b):
                delta_i, counter = 1, 1
                while delta_i < rng_matrix:
                    if b:
                        is_line = edge_matrix(arr, i, j+delta_i, b)
                    else:
                        is_line = edge_matrix(arr, i+delta_i, j, b)
                    if is_line:
                        counter = counter + 1
                    delta_i = delta_i + 1
                # enough tests are pass, its not a noise - probably rectangle!
                if counter >= rng_matrix*RNG_ERR_RECT:
                    # verify rectangle
                    if is_rect(i, j, arr, b):
                        # back to original coordinates
                        if b:
                            return j, i+1
                        return i, j+1


# the tests for being edge(each i, j)
def edge_matrix(arr, i, j, b):
    if b:
        tmp = i
        i, j = j, tmp
    if arr[i][j] == 0:
        if arr[i][j + 1] == 1:
            return True
    return False


# rectangle side is defined up, 21 pixels for each edge and 0's wrap in each direction
def is_rect(i, j, arr, b):
    if b:
        tmp = i
        i, j = j, tmp
    counter = 0
    for x in range(EDGE_SIZE):
        for y in range(EDGE_SIZE):
            if arr[i+x][j+y] == rectangle_side[x][y]:
                counter = counter+1
                # enough witnesses for rectangle
                if counter > NUM_OF_PIX_RECT*RNG_ERR_RECT:
                    return True
    return False


# defining the align coordinates and transform  with perspective projection
def get_perspective(img, cords):
    bases = create_base([[(0, 0), cords[0]], [(w, 0), cords[1]], [(w, h), cords[2]], [(0, h), cords[3]]])
    img = img.transform((w, h), method=Image.PERSPECTIVE, data=bases)
    return img


# create base co-effitions for the transform
def create_base(coordinates):
    arr_to_a, arr_to_b = [], []
    for pair in coordinates:
        first, sec = return_arrays(pair[0], pair[1])
        arr_to_a.append(first)
        arr_to_a.append(sec)
        for ind in range(2):
            arr_to_b.append(pair[1][ind])
    a = np.array(arr_to_a, dtype=np.float32)
    b = np.array(arr_to_b, dtype=np.float32)
    return linalg.solve(a, b)


# manipulation for convenience
def return_arrays(a, b):
    a0, a1, b0, b1 = a[0], a[1], b[0], b[1]
    first = [a0, a1, 1, 0, 0, 0, -b0*a0, -b0*a1]
    second = [0, 0, 0, a0, a1, 1, -b1*a0, -b1*a1]
    return first, second


# if the image was scan in the opposed direction
def check_opp(aligned):
    aligned_data = np.array(aligned)
    b = opp(aligned_data)
    if not b:
        aligned = aligned.rotate(FULL_ROUND)
        aligned_data = np.array(aligned)
    return aligned_data, aligned


# the differentiation between opposed scan and straight one
def opp(data):
    i = PRINT_BEG
    while i < PRINT_BEG_MAX:
        for j in RANGE_MID:
            if is_black(data[i][j]):
                return True
        i = i+1
    return False

def recognize_chars(data):
    count = 0
    first_black = True
    for i in range(1700, h):
        for j in range(400, 1100):
            if is_black(data[i][j]):
                if first_black:
                    x,y = j, i
                    first_black = False
                count = count+1
    if count==0:
        return 'a'
    elif count<40:
        return 'l'
    elif count<60:
        return i_or_j(x, y, data)
    elif count<75:
        return 'k'
    elif count<90:
        return d_or_f(x, y, data)
    elif count<100:
        return e_or_b(x, y, data)
    elif count<140:
        return 'h'
    else:
        return 'm'


def i_or_j(x,y, data):
    min_x = w
    max_x = 0
    for m in range(y, h):
        for n in range(x-10, x+10):
            if is_black(data[m][n]):
                if n<min_x:
                    min_x = n
                elif n>max_x:
                    max_x = n
    identifier = max_x - min_x
    if identifier>4:
        return 'j'
    return 'i'


def d_or_f(x,y, data):
    black_rows = 0
    for i in range(y, h):
        count_in_row = 0
        for j in range(x, x+50):
            if is_black(data[i][j]):
                count_in_row = count_in_row+1
        if count_in_row>8:
            black_rows = black_rows+1
    if black_rows>4:
        return 'f'
    return 'd'


def e_or_b(x,y,data):
    black_rows = 0
    for i in range(y, h):
        count_in_row = 0
        for j in range(x, x+50):
            if is_black(data[i][j]):
                count_in_row = count_in_row+1
        if count_in_row>3:
            black_rows = black_rows+1
    if black_rows>10:
        return 'b'
    return 'e'

def from_char_to_term(char_symbol):
    switcher = {'a': term_A, 'b': term_B, 'c':term_C, 'd':term_D, 'e':term_E, 'f':term_F, 'g':term_G, 'h':term_H, 'i':term_I,
                'j':term_J, 'k':term_K, 'l':term_L, 'm':term_M }
    return switcher.get(char_symbol)

def find_lines_words_component(data, term):
    aligned_data = np.copy(data)
    # raise the blue channel in the matrix in order to prepare the matrix to binarization.
    aligned_data = raise_blue(aligned_data)
    # delete the skeleton between boxes
    aligned_data = between_lines(aligned_data)
    # we need to render the img after all manipulation
    aligned = Image.fromarray(aligned_data, 'RGB')
    # grayscale and binarization
    gray = aligned.convert('L')
    bw = gray.point(lambda x: 0 if x < 200 else 255, '1')
    bin_data = np.array(bw)
    # finding bounds in y axis of each written line of input
    written_ranges = find_written_rows(bin_data)
    # now with each range of input we find connected components in order to analyze each component
    sides, components = connected_components(bin_data, written_ranges)
    # detecting word by largest distances between components in x axis
    words, words_ind = find_words(components, term)
    return written_ranges, sides, components, words, words_ind


# running over all img, and raising the blue channel in each i,j: in order to "ignore" yellow lines
def raise_blue(data):
    for i in range(h):
        for j in range(w):
            data[i][j][2] = MAX_CHAN
    return data


def ignore_large_noises(data):
    for i in range(h):
        for j in range(w):
            if not np.array_equal(data[i][j], WHITE_RGB):
                if data[i][j][0]>200:
                    if data[i][j][1]>200:
                        data[i][j] = WHITE_RGB

    return data


# delete printed (skeleton)
def between_lines(data):
    # delete up the yellows
    data = delete_colors(data, range(boxes[0][0]), range(w), [0], [False], [RNG_MAX])
    for x in range(2):
        # delete red(pink) skeleton
        data = delete_colors(data, range(boxes[x][1], boxes[x+1][0]), range(MID_OF_IMG, w), [1, 0], [False, True],
                             [240, 50])
        # delete black skeleton
        data = delete_colors(data, range(boxes[x][1] + PADDING_DOWN_LINE, boxes[x+1][0]),  range(w), [1], [False],
                             [RNG_MAX])
    # delete only the printed(from the mid and right)
    for x in range(11):
        data = delete_colors(data, range(boxes[x][1] + PADDING_DOWN_LINE, boxes[x+1][0] + RNG_ERROR), range(MID_OF_IMG + DIF_REG_LINES,
                                                                                                            w),
                             [0], [False], [RNG_MAX])
    # last line - printed thrugh hall width
    data = delete_colors(data, range(boxes[11][1] + PADDING_DOWN_LINE, boxes[12][0]), range(w), [0], [False], [RNG_MAX])
    return data


# method that actually paint the pixel in white if needed
def delete_colors(data, rng_i, rng_j, ind_rgb, is_bigger_than, lhs):
    for i in rng_i:
        for j in rng_j:
            cont = True
            for rgb in range(len(ind_rgb)):
                ind_to_check = data[i][j][ind_rgb[rgb]]
                if is_bigger_than[rgb]:
                    that_channel = (ind_to_check > lhs[rgb])
                else:
                    that_channel = (ind_to_check < lhs[rgb])
                cont = cont and that_channel
            if cont:
                data[i][j] = WHITE_RGB
    return data


# find the range of written rows
def find_written_rows(bin_data):
    written_rows = []
    # not in down black rectangles
    for i in range(h - 30):
        count_black = 0
        for j in range(w):
            if not bin_data[i][j]:
                count_black = count_black + 1
        # if pixels are written in this rows: add the row to written rows data structure
        if count_black > 0:
            written_rows.append(i)
    return bound_written(written_rows)


def bound_written(written_rows):
    ranges = []
    i = 0
    while i < len(written_rows) - 1:
        first = written_rows[i]
        i = i+1
        while i < len(written_rows) - 1 and (written_rows[i] - written_rows[i-1]) < 2:
            i = i+1
        last = written_rows[i - 1]
        ranges.append([first, last])
    rng_no_noise = []
    # delete rows that marked, but are just noise:
    for rng in ranges:
        if rng[1] - rng[0] > avg_letter_size:
            rng_no_noise.append(rng)
    # padding:
    for ind in range(len(rng_no_noise)):
        rng_no_noise[ind][0] = rng_no_noise[ind][0] - PADD
        rng_no_noise[ind][1] = rng_no_noise[ind][1] + PADD
    return rng_no_noise


# side effect method
def paint_lines(ranges, data):
    for rng in ranges:
        for j in range(w):
            data[rng[0]][j] = [200,0,15]
            data[rng[1]][j] = [0,200,15]
    return data


# this method send each row in iterative way to the component search
def connected_components(bin_data, rows):
    sided = []
    components = []
    for row in rows:
        side, components_row = find_connected_components(row, bin_data)
        sided.append(side)
        components.append(components_row)
    return sided, components


def find_connected_components(row, bin_data):
    connected_components_in_curr_row = []
    sze = row[1] - row[0]
    # maintaining array of components, not used yet
    zrs = np.zeros((sze, w), dtype=int)
    j = w - 1
    count_comp = 1
    # j decreases from convenience reasons - hebrew is written from right to left
    while j > 0:
        i = row[0]
        while i < row[1]:
            if not bin_data[i][j]:
                visited, zrs = dfs(i, j, bin_data, zrs, [[i, j]], count_comp, row[0])
                if visited is not None:
                    next_j = find_min_j(visited, j)
                    # if its not noise: add the bound coordinates to the data structure
                    if (j - next_j) > 3:
                        curr_comp = [[find_max_j(visited, j), next_j],find_min_max_i(visited, i, i)]
                        connected_components_in_curr_row.append(curr_comp)
                        # next component more possibly to start after this, but this is for not getting into infinite
                        # loop or find the same component several times.
                        # anyway, find_max_j find the start of next component even if it start into current component
                        # interval
                        i = curr_comp[1][1]
                        while i < row[1]:
                            for m in range(next_j, j):
                                if not bin_data[i][m]:
                                    if not visited_pixel(i,m,visited):
                                        n_visites, zrs = dfs(i, m, bin_data, zrs, [[i, m]], count_comp, row[0])
                                        # enough pixels to determine component
                                        if len(n_visites) > 2:
                                            connected_components_in_curr_row.append([[find_max_j(n_visites, m),
                                                                                      find_min_j(n_visites, j)],
                                                                                     find_min_max_i(n_visites, i, i)])
                                            i = row[1]
                                            break
                            i = i + 1
                        i = row[0]
                        j = next_j
                        count_comp = count_comp+1
            i = i+1
        j = j - 1
    co = connected_components_in_curr_row
    row_side = [co[0][0][0] + PADD, co[len(co)-1][0][1]-PADD]
    return row_side, co


# in order to pass the hall component, using the dfs recursive familiar algorithm
def dfs(i, j, data, zrs, visited, counter, start_i):
    zrs[i-start_i][j] = counter
    # where to look for continuation
    neibors = [[i+1, j], [i-1, j], [i, j+1], [i, j-1], [i+1, j+1], [i+1, j-1], [i-1, j+1], [i-1, j-1]]
    for neib in neibors:
        if 0 < (neib[0] - start_i) < len(zrs) and 0 < neib[1] < w and not data[neib[0]][neib[1]]:
            if not visited_pixel(neib[0], neib[1], visited):
                # memoization data structure
                visited.append([neib[0], neib[1]])
                # recursive call
                visited, zrs = dfs(neib[0], neib[1], data, zrs, visited, counter, start_i)
    return visited, zrs


# memoization
def visited_pixel(i, j, visited):
    for pix in visited:
        if i == pix[0] and j == pix[1]:
            return True
    return False


# most left coord
def find_min_j(visited, min_j):
    for pix in visited:
        if min_j > pix[1]:
            min_j = pix[1]
    return min_j - PADD


# most right coord
def find_max_j(visited, max_j):
    for pix in visited:
        if max_j < pix[1]:
            max_j = pix[1]
    return max_j + PADD


# highest and lowest
def find_min_max_i(visited, min_i, max_i):
    for pix in visited:
        if min_i > pix[0]:
            min_i = pix[0]
        if max_i < pix[0]:
            max_i = pix[0]
    return [min_i, max_i]


def find_words(components, term):
    spaces = []
    words = []
    words_inds = []
    for i in range(2, min(len(components), 13)):
        curr_line = components[i]
        # the first component from right, for saving half of memory acsess
        cur_comp = curr_line[0]
        curr_spaces = []
        for j in range(len(components[i])-1):
            curr_c_x = cur_comp[0]
            curr_end = curr_c_x[1]
            next_comp = curr_line[j+1]
            next_c_x = next_comp[0]
            next_start = next_c_x[0]
            curr_space = curr_end-next_start
            curr_spaces.append(curr_space)
            cur_comp = next_comp
        space_rec = find_larger_spaces(curr_spaces, i, term)
        spaces.append(space_rec)
        words_ind = spaces_to_words(space_rec, len(components[i])-1)
        words_curr = []
        for word in words_ind:
            first = word[0]
            last = word[1]
            start_j = components[i][first][0][0]
            end_j = components[i][last][0][1]
            highest = h
            lowest = 0
            for x in range(first, last+1):
                comp_high = components[i][x][1][0]
                comp_low = components[i][x][1][1]
                if comp_high< highest:
                    highest = comp_high
                if comp_low > lowest:
                    lowest = comp_low
            words_curr.append([[start_j, end_j],[highest, lowest]])
        words.append(words_curr)
        words_inds.append(words_ind)
    return words,words_inds


def spaces_to_words(space_rec, last_comp):
    words = []
    first = 0
    for bound in space_rec:
        words.append([first, bound])
        first = bound+1
    words.append([first, last_comp])
    return words


# largest distances between components are probably spaces
def find_larger_spaces(spaces, i, term):
    num_of_spaces = term[i-2].count(" ")
    max_indexes = []
    for s in range(num_of_spaces):
        max_space = max(spaces)
        ind = spaces.index(max_space)
        spaces[ind] = 0
        max_indexes.append(ind)

    max_indexes.sort()
    return max_indexes


def str_coords(left, right, up, down):
    r_num = int(right)
    if r_num>=w:
        r_num = w-1
        right = str(r_num)
    ret = pair_str(left,up)+" "+pair_str(left, down)+" "+pair_str(right, down)+" "+pair_str(right,up)
    return ret


def pair_str(x,y):
    return x+","+y


def xml(rows, name, sides, components, words, words_ind, term):

    root = ET.Element("PcGts" )
    root.set("xsi:schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15/pagecontent.xsd")
    root.set("xmlns","http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15")
    root.set("xmlns:xsi","http://www.w3.org/2001/XMLSchema-instance")
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Coral Burg"
    curr_date = str(datetime.datetime.now())
    [date1, date2] = curr_date.split()
    [hours, mili] = date2.split('.')
    curr_date = date1+"T"+hours
    ET.SubElement(metadata, "Created").text = curr_date
    ET.SubElement(metadata, "LastChange").text = curr_date
    page = ET.SubElement(root, "Page")
    page.set("imageFilename",name)
    page.set("imageWidth", str(w))
    page.set("imageHeight", str(h))
    page.set("readingDirection","right-to-left")
    page.set("primaryLanguage", "Hebrew")
    reg = ET.SubElement(page, "TextRegion", primaryLanguage="Hebrew", readingDirection="right-to-left", primaryScript="Hebr - Hebrew",
                        type= "paragraph", id = "r"+str(0))
    count_component =0
    count_word = 0
    ET.SubElement(reg, "Coords", points=str_coords(str(1),str(w-1), str(1), str(h-5)))
    # possibly more than 13 lines - wrong input
    if len(rows)<13:
        row_len = len(rows)
    else:
        row_len = 13

    for counter_lines in range(row_len):
        txt_line = ET.SubElement(reg, "TextLine", id="l"+str(counter_lines), primaryLanguage="Hebrew"
                                 , readingDirection="right-to-left")
        up, down = str(rows[counter_lines][0]), str(rows[counter_lines][1])
        right, left = str(sides[counter_lines][0]), str(sides[counter_lines][1])
        ET.SubElement(txt_line, "Coords", points = str_coords(left, right, up, down))
        sen = term[counter_lines-2]
        #   the first and second lines are spacial case, handled next
        if 2<=counter_lines<=13:
            line = counter_lines-2
            arr_of_words = sen.split(' ')
            equal_num = len(arr_of_words)==len(words[line])
            for i in range(len(words[line])):
                word = ET.SubElement(txt_line, "Word", id = "w"+str(count_word))
                count_word = count_word+1
                curr_word = words[line][i]
                up_w, down_w = str(curr_word[1][0]), str(curr_word[1][1])
                right_w, left_w = str(curr_word[0][0]), str(curr_word[0][1])
                indxs = words_ind[line][i]
                start = indxs[0]
                end = indxs[1]
                ET.SubElement(word, "Coords", points=str_coords(left_w, right_w, up_w, down_w))

                for count_components in range(start, end+1):
                    component = ET.SubElement(word, "Glyph", id="c"+str(count_component))
                    count_component = count_component+1
                    curr_comp = components[counter_lines][count_components]
                    up_c, down_c = str(curr_comp[1][0]), str(curr_comp[1][1])
                    right_c,left_c = str(curr_comp[0][0]), str(curr_comp[0][1])
                    ET.SubElement(component, "Coords", points=str_coords(left_c, right_c, up_c, down_c))
                    comp_eq = ET.SubElement(component, "TextEquiv")
                    ET.SubElement(comp_eq, "Unicode")
                if equal_num:
                    txt_eq_w = ET.SubElement(word, "TextEquiv")
                    printed_word = arr_of_words[i]
                    ET.SubElement(txt_eq_w, "Unicode").text = printed_word
            txt_eq = ET.SubElement(txt_line, "TextEquiv")
            ET.SubElement(txt_eq, "Unicode").text = sen
        #   two firsts lines are only components but a word was needed for format - spacial case
        else:
            word_line = ET.SubElement(txt_line, "Word", id="w" + str(count_word))
            count_word = count_word+1
            ET.SubElement(word_line, "Coords", points=str_coords(left, right, up, down))
            for count_components in range(len(components[counter_lines])):
                component = ET.SubElement(word_line, "Glyph", id="c" + str(count_component))
                count_component = count_component+1
                curr_comp = components[counter_lines][count_components]
                up_c, down_c = str(curr_comp[1][0]), str(curr_comp[1][1])
                right_c, left_c = str(curr_comp[0][0]), str(curr_comp[0][1])
                ET.SubElement(component, "Coords", points=str_coords(left_c, right_c, up_c, down_c))
                comp_eq = ET.SubElement(component, "TextEquiv")
                ET.SubElement(comp_eq, "Unicode")
            text_word = ET.SubElement(word_line, "TextEquiv")
            ET.SubElement(text_word, "Unicode")
            line_eq = ET.SubElement(txt_line, "TextEquiv")
            ET.SubElement(line_eq, "Unicode")

    text_reg = ET.SubElement(reg, "TextEquiv")
    ET.SubElement(text_reg, "Unicode")
    tree = ET.ElementTree(root)
    return tree


#   each image in current directory manipulated by the main function that has a side effect:
#   create a new directory in the current one that contains 2 directories: input and output. the input contains
#   the aligned images, and the output contains the xml files for each image by the format with all data that we need.
def main():
    os.mkdir("input_output")
    curr_dir = os.getcwd()
    new_dir = curr_dir+"/input_output"
    os.chdir(new_dir)
    os.mkdir("input")
    os.mkdir("output")
    os.chdir(curr_dir)
    for file in os.listdir("."):
        if file.endswith(".jpg"):
            inputs, output = main_func(file)
            os.chdir(new_dir+"/input")
            inputs.save(file)
            os.chdir(new_dir+"/output")
            str_name = file[:-3]
            output.write(str_name + "xml",  encoding="UTF-8", xml_declaration=True)
            os.chdir(curr_dir)


if __name__ == '__main__':
        main()
