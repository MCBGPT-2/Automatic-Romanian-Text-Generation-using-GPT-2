import os, re

conf = {
    'ROOT': os.path.dirname(os.path.realpath(__file__ + '/../../../../../../')) + os.sep,
    'CONFIG_PATH':  os.path.dirname(os.path.realpath(__file__ + '/../../../../../../')) + os.sep + 'config' + os.sep + 'utils',
    'APP_DIR': 'src',
    'APP': os.path.dirname(
        os.path.realpath(__file__ + '/../..')) + os.sep + 'src' + os.sep + 'path to application',
    'PYTHON_DIR': os.path.dirname(
        os.path.realpath(
            __file__ + '/../../../../../../')) + os.sep + 'src' + os.sep + 'path to application',
    'site_name': 'Site name',
    'base_url': 'Site base url',
    'dbHost': 'localhost',
    'dbName': 'database',
    'dbUser': 'username',
    'dbPassword': 'passwd',
    'stopWords': 'stop_words.txt',
    'pozitiveWords': 'positiveWords.txt',
    'negativeWords': 'negativeWords.txt',
    'emoticonsWords': 'emoticons.txt',
    'notSimilarWords': 'not_similar_words.txt',
}


class Config(object):
    def __init__(self):
        self.config = conf

    def get_property(self, property_name):
        if property_name not in self.config.keys():
            return 'Is not in the config file'
        return self.config[property_name]

    def clean_text(cls, phrases):
        regex_string = [
            {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
            {r'\s+': u' '},  # replace consecutive spaces
            {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
            {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>
            {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>
            {r'<head>.*<\s*(/head|body)[^>]*>': u' '},  # remove <head> to </head>
            {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
            {r'[ \t]*<[^<]*?/?>': u' '},  # remove remaining tags
            {r'^\s+': u' '}  # remove spaces at the beginning
        ]
        for rule in regex_string:
            for (k, v) in rule.items():
                regex = re.compile(k)
                phrases = regex.sub(v, phrases)

        phrases = phrases.rstrip()
        phrases = phrases.replace(".", " ")
        phrases = phrases.replace("--", "-")
        phrases = phrases.replace("--", " ")
        phrases = phrases.replace("_", " ")
        phrases = phrases.replace(",", " ")
        phrases = phrases.replace("@", " ")
        phrases = phrases.replace("#", " ")
        phrases = phrases.replace("$", " ")
        phrases = phrases.replace("%", " ")
        phrases = phrases.replace("^", " ")
        phrases = phrases.replace("&", "and")
        phrases = phrases.replace("*", " ")
        phrases = phrases.replace("(", " ")
        phrases = phrases.replace(")", " ")
        phrases = phrases.replace("+", " ")
        phrases = phrases.replace("=", " ")
        phrases = phrases.replace("-", " ")
        phrases = phrases.replace("?", " ")
        phrases = phrases.replace("!", " ")
        phrases = phrases.replace("\'", " ")
        phrases = phrases.replace("\"", " ")
        phrases = phrases.replace("{", " ")
        phrases = phrases.replace("}", " ")
        phrases = phrases.replace("[", " ")
        phrases = phrases.replace("]", " ")
        phrases = phrases.replace("<", " ")
        phrases = phrases.replace(">", " ")
        phrases = phrases.replace("~", " ")
        phrases = phrases.replace("`", " ")
        phrases = phrases.replace(":", " ")
        phrases = phrases.replace(";", " ")
        phrases = phrases.replace("|", " ")
        phrases = phrases.replace("(", " ")
        phrases = phrases.replace(")", " ")
        phrases = phrases.replace("&nbsp;", " ")
        phrases = phrases.replace("\\", " ")
        phrases = phrases.replace("/", " ")
        phrases.lower()
        phrases = ''.join([i for i in phrases if not i.isdigit()])
        return phrases

