"""

"""


import sublime, sublime_plugin
from os import path
import operator
import sys
import re

BASE_FILE = path.abspath(__file__)
BASE_PATH = path.dirname(BASE_FILE)
PGEEK_SETTINGS = "proseGeek.sublime-settings"

nltk_path = path.join(BASE_PATH, r"nltk")
bs4_path = path.join(BASE_PATH, r"bs4")

sys.path.append(BASE_PATH)
sys.path.append(nltk_path)
sys.path.append(bs4_path)

from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.corpus import cmudict 
from nltk import collocations
from nltk import probability
from nltk import tokenize
from nltk import metrics

cmud = cmudict.dict() 


def count_syllables(word, ref_dict):
    syllables=[]
    try:
        pronounciation = ref_dict[word.lower()]
        for syllable in pronounciation[0]:
            if str.isdigit(syllable[-1]):
                syllables.append(syllable)
        return len(syllables)
    except KeyError:
        return(0)


class pgReportTemplate(object):
	def __init__(self):
		self.template =r'''# Prose Geek Report: {sourceview_name}

## Counts
* **Total Characters:** {count_chars}
* **Total Words:** {count_words}
* **Unique Words:** {count_stemmed}
* **Total Sentences:** {count_sentences}

## Averages:
* **Characters per word:** {av_char_word:0.1f}
* **Words per sentence:** {av_words_sentence:0.1f}
* **Lexical diversity (average usage of each word):** {lexical_diversity:0.1f}

## Outliers:
* **Longest Word:** {long_word}
* **Shortest Sentece:**
>*{short_sentence}*

* **Longest Sentece:**
>*{long_sentence}*


## Frequencies
### Top {top_word_count} frequently used words
{fd_report}

--- 

### Top {top_bigrams_count} frequently used 2-word phrases (bi-grams)
{bigram_report}

--- 

### Top {top_trigrams_count} frequently used 3-word phrases (tri-grams)
{trigram_report}

--- 
''' 


	def output(self):
		return self.template



class proseGeek(object):
	"""docstring for proseGeek

	Arguments:

	raw_text: (string) Text to be processed

	"""

	def __init__(self, raw_text):
		self.raw_text = raw_text

		# Load settings
		self.external_settings = sublime.load_settings(PGEEK_SETTINGS)

		# Default values if an external setting is missing
		self.active_settings={}
		default_values={
			"filter_stopwords" : True,
			"stopwords_filepath" : path.join(BASE_PATH, "default_stopwords.txt"),
			"strip_html" : True,
			"strip_markdown" : True,
			"top_word_count" : 20,
			"top_bigrams_count" : 10,
			"top_trigrams_count" : 10,
			"collocation_filter": 3
		}
		for setting in default_values:
			external_setting = self.external_settings.get(str(setting))
			if external_setting is None:
				self.active_settings[setting] = default_values[setting]
			else:
				self.active_settings[setting] = external_setting

		# Import Stopwords
		with open(self.active_settings["stopwords_filepath"], "r") as f:
			self.stopwords = f.read().splitlines()

	def strip_markup(self):
		clean_text = self.raw_text
		
		# Parse out HTML
		if self.active_settings["strip_html"] == True:
			soup = BeautifulSoup(self.raw_text, "html.parser")

			# Strip the Javascript and CSS style blocks
			for element in soup(["script", "style"]):
				element.decompose()

			# This is a hack, without it, some pages are parsed to appear as blank. No idea why this happens 
			soup = BeautifulSoup(str(soup), "html.parser")
			nohtml_text = soup.get_text()

			# Strip needless whitespace characters
			lines = [line.strip() for line in nohtml_text.splitlines()]
			nomultispace_lines = [piece.strip() for line in lines for piece in line.split("  ")]
			clean_text = '\n'.join(piece for piece in nomultispace_lines if piece)


		# Strip Markdown
		if self.active_settings["strip_markdown"] == True:
			md_re_pattern = r"[\*#`~_\|>]+|-[-]+|\[.+\]:*|\(http.+\)|http.+"
			clean_text = re.sub(md_re_pattern, "", clean_text)

		self.clean_text = clean_text


	def process(self):
		# Tokenize sentences
		self.all_sents = tokenize.sent_tokenize(self.clean_text)
		# Tokenize words and punctuation
		self.all_tokens = tokenize.word_tokenize(self.clean_text)
		# Tokenize words only
		tokenizer = tokenize.RegexpTokenizer("[\w']+")
		self.all_words = tokenizer.tokenize(self.clean_text)
		

		# Filter out the stopwords
		self.stopfil_words = [word for word in self.all_words if word.lower() not in self.stopwords]

		# lowercase
		self.all_words_lc = [word.lower() for word in self.all_words]
		self.stopfil_words_lc = [word.lower() for word in self.stopfil_words]
		
		# Get vocabulary
		self.vocab = sorted(set([word.lower() for word in self.stopfil_words]))


	def basic_stats(self):
		self.count_chars = len(self.raw_text)
		self.count_words = len(self.all_words)
		self.count_sentences = len(self.all_sents)
		self.av_char_word = self.count_chars/self.count_words
		self.av_words_sentence = self.count_words/self.count_sentences
		self.count_vocab = len(self.vocab)

		porter_stemmer = PorterStemmer()
		self.stemmed_words = [porter_stemmer.stem(word) for word in self.stopfil_words]
		self.stemmed_vocab = sorted(set(self.stemmed_words))
		self.count_stemmed = len(self.stemmed_vocab)
		self.lexical_diversity = self.count_words/self.count_stemmed

		self.long_sentence = " ".join(sorted(self.all_sents, key=lambda w: len(w), reverse=True)[0].splitlines())
		self.short_sentence = " ".join(sorted(self.all_sents, key=lambda w: len(w), reverse=False)[0].splitlines())
		self.long_word = sorted(self.all_words, key=lambda w: len(w), reverse=True)[0]



	def top_words(self):
		# Frequency distribution of most common words
		self.fd = probability.FreqDist(self.stopfil_words)
		fdc = self.fd.most_common(self.active_settings["top_word_count"])
		return fdc


	def ngrams(self):
		# Bigrams
		bgfinder = collocations.BigramCollocationFinder.from_words(self.all_words_lc)
		bgfinder.apply_freq_filter(self.active_settings["collocation_filter"]) 
		bigrams_prefiltered = bgfinder.ngram_fd.most_common()

		self.bigrams = []
		# exclude combinations where both words are in the stopword list
		for pair, freq in bigrams_prefiltered:
			if len(self.bigrams) >= self.active_settings["top_bigrams_count"]:
				break
			else:
				if not((pair[0] in self.stopwords) and (pair[1] in self.stopwords)):
					self.bigrams.append((pair, freq))

		# Trigrams
		tgfinder = collocations.TrigramCollocationFinder.from_words(self.all_words_lc)
		tgfinder.apply_freq_filter(self.active_settings["collocation_filter"])
		trigrams_prefiltered = tgfinder.ngram_fd.most_common()
		
		self.trigrams = []
		# exclude combinations where three words are in the stopword list
		for pair, freq in trigrams_prefiltered:
			if len(self.trigrams) >= self.active_settings["top_trigrams_count"]:
				break
			else:
				if not((pair[0] in self.stopwords) and (pair[1] in self.stopwords) and (pair[2] in self.stopwords)):
					self.trigrams.append((pair, freq))




class pgeekReportCommand(sublime_plugin.WindowCommand):
	def run(self):
		# Set the source view
		if self.window.active_view():
			sourcePage = self.window.active_view()
		else:
			return
		
		#If no selection, or multiple selections, get all content. Otherwise only get the content from the selection
		regions = sourcePage.sel()

		# If there is only one region, and it is not empty (single caret), use that region, otherwise use everything
		if (len(regions) == 1) and (not regions[0].empty()):
			contentRegion = regions[0] 
		else:
			contentRegion = sublime.Region(0, sourcePage.size())

		raw_viewContent = sourcePage.substr(contentRegion)

		# Create a proseGeek object with the content of the region or view.
		geek = proseGeek(raw_viewContent)
		geek.strip_markup()
		geek.process()
		geek.basic_stats()
		geek.ngrams()

		# Create a new view for the report
		try:
			sourceview_name = path.split(self.window.active_view().file_name())[1]
		except:
			sourceview_name = "unsaved_file"

		# Build the report for output
		templateObject = pgReportTemplate()
		template = templateObject.output()
		
		# Tabulate data
		# 		Topwords
		topwords = geek.top_words()
		fdr = "|Rank|Word|Count|\n|---|---|---|\n"
		if len(topwords) < 1:
			fdr += "||||"
		for count, (word, freq) in enumerate(topwords):
			fdr += "|{0}.|{1}|{2}|\n".format(count+1, word, freq)

		# 		Bigrams
		bigrams = geek.bigrams
		bigram_report = "|Rank|Bi-gram|Count|\n|---|---|---|\n"
		if len(bigrams) < 1:
			bigram_report += "||||"
		for count, (gram, freq) in enumerate(bigrams):
			bigram_report += "|{0}.|{1} {2}|{3}|\n".format(count+1, gram[0], gram[1], freq)

		# 		Trigrams
		trigrams = geek.trigrams
		trigram_report = "|Rank|Tri-gram|Count|\n|---|---|---|\n"
		if len(trigrams) < 1:
			trigram_report += "||||"
		for count, (gram, freq) in enumerate(trigrams):
			trigram_report += "|{0}.|{1} {2} {3}|{4}|\n".format(count+1, gram[0], gram[1], gram[2], freq)

		report = template.format(
			sourceview_name=sourceview_name,
			top_word_count=geek.active_settings["top_word_count"],
			fd_report=fdr,
			count_chars=geek.count_chars,
			count_words=geek.count_words,
			count_sentences=geek.count_sentences,
			av_words_sentence=geek.av_words_sentence,
			av_char_word=geek.av_char_word,
			count_stemmed=geek.count_stemmed,
			lexical_diversity=geek.lexical_diversity,
			top_bigrams_count=geek.active_settings["top_bigrams_count"],
			top_trigrams_count=geek.active_settings["top_trigrams_count"],
			bigram_report=bigram_report,
			trigram_report=trigram_report,
			long_word=geek.long_word,
			short_sentence=geek.short_sentence,
			long_sentence=geek.long_sentence
			)

		# Create and populate the new view
		reportPage = self.window.new_file()
		reportPage.set_scratch(True)
		reportPage.set_name("ProseGeek Report: _{0}_.md".format(sourceview_name))
		reportPage.run_command("pgeek_view_update", {"content":report})
		sublime.active_window().focus_view(reportPage)

		# Cleanup
		del geek, templateObject


class pgeekViewUpdateCommand(sublime_plugin.TextCommand):
	def run(self, edit, content):
		self.view.replace(edit, sublime.Region(0, self.view.size()), content)
