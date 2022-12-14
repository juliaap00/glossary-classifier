# import the necessary libraries
import nltk
import string
import re
import os
special_char = "¿”“'ª»«...—‘’º"
path = './corpus'
preprocessing_path = "./processed-corpus"
stopWords = ["actualmente","acuerdo","adelante","ademas","además","adrede","afirmó","agregó","ahi","ahora","ahí","al","algo","alguna","algunas","alguno","algunos","algún","alli","allí","alrededor","ambos","ampleamos","antano","antaño","ante","anterior","antes","apenas","aproximadamente","aquel","aquella","aquellas","aquello","aquellos","aqui","aquél","aquélla","aquéllas","aquéllos","aquí","arriba","arribaabajo","aseguró","asi","así","atras","aun","aunque","ayer","añadió","aún","b","bajo","bastante","bien","breve","buen","buena","buenas","bueno","buenos","c","cada","casi","cerca","cierta","ciertas","cierto","ciertos","cinco","claro","comentó","como","con","conmigo","conocer","conseguimos","conseguir","considera","consideró","consigo","consigue","consiguen","consigues","contigo","contra","cosas","creo","cual","cuales","cualquier","cuando","cuanta","cuantas","cuanto","cuantos","cuatro","cuenta","cuál","cuáles","cuándo","cuánta","cuántas","cuánto","cuántos","cómo","d","da","dado","dan","dar","de","debajo","debe","deben","debido","decir","dejó","del","delante","demasiado","demás","dentro","deprisa","desde","despacio","despues","después","detras","detrás","dia","dias","dice","dicen","dicho","dieron","diferente","diferentes","dijeron","dijo","dio","donde","dos","durante","día","días","dónde","e","ejemplo","el","ella","ellas","ello","ellos","embargo","empleais","emplean","emplear","empleas","empleo","en","encima","encuentra","enfrente","enseguida","entonces","entre","era","erais","eramos","eran","eras","eres","es","esa","esas","ese","eso","esos","esta","estaba","estabais","estaban","estabas","estad","estada","estadas","estado","estados","estais","estamos","estan","estando","estar","estaremos","estará","estarán","estarás","estaré","estaréis","estaría","estaríais","estaríamos","estarían","estarías","estas","este","estemos","esto","estos","estoy","estuve","estuviera","estuvierais","estuvieran","estuvieras","estuvieron","estuviese","estuvieseis","estuviesen","estuvieses","estuvimos","estuviste","estuvisteis","estuviéramos","estuviésemos","estuvo","está","estábamos","estáis","están","estás","esté","estéis","estén","estés","ex","excepto","existe","existen","explicó","expresó","f","fin","final","fue","fuera","fuerais","fueran","fueras","fueron","fuese","fueseis","fuesen","fueses","fui","fuimos","fuiste","fuisteis","fuéramos","fuésemos","g","general","gran","grandes","gueno","h","ha","haber","habia","habida","habidas","habido","habidos","habiendo","habla","hablan","habremos","habrá","habrán","habrás","habré","habréis","habría","habríais","habríamos","habrían","habrías","habéis","había","habíais","habíamos","habían","habías","hace","haceis","hacemos","hacen","hacer","hacerlo","haces","hacia","haciendo","hago","han","has","hasta","hay","haya","hayamos","hayan","hayas","hayáis","he","hecho","hemos","hicieron","hizo","horas","hoy","hube","hubiera","hubierais","hubieran","hubieras","hubieron","hubiese","hubieseis","hubiesen","hubieses","hubimos","hubiste","hubisteis","hubiéramos","hubiésemos","hubo","i","igual","incluso","indicó","informo","informó","intenta","intentais","intentamos","intentan","intentar","intentas","intento","ir","j","junto","k","l","la","lado","largo","las","le","lejos","les","llegó","lleva","llevar","lo","los","luego","lugar","m","mal","manera","manifestó","mas","mayor","me","mediante","medio","mejor","mencionó","menos","menudo","mi","mia","mias","mientras","mio","mios","mis","misma","mismas","mismo","mismos","modo","momento","mucha","muchas","mucho","muchos","muy","más","mí","mía","mías","mío","míos","n","nada","nadie","ni","ninguna","ningunas","ninguno","ningunos","ningún","no","nos","nosotras","nosotros","nuestra","nuestras","nuestro","nuestros","nueva","nuevas","nuevo","nuevos","nunca","o","ocho","os","otra","otras","otro","otros","p","pais","para","parece","parte","partir","pasada","pasado","paìs","peor","pero","pesar","poca","pocas","poco","pocos","podeis","podemos","poder","podria","podriais","podriamos","podrian","podrias","podrá","podrán","podría","podrían","poner","por","por qué","porque","posible","primer","primera","primero","primeros","principalmente","pronto","propia","propias","propio","propios","proximo","próximo","próximos","pudo","pueda","puede","pueden","puedo","pues","q","qeu","que","quedó","queremos","quien","quienes","quiere","quiza","quizas","quizá","quizás","quién","quiénes","qué","r","raras","realizado","realizar","realizó","repente","respecto","s","sabe","sabeis","sabemos","saben","saber","sabes","sal","salvo","se","sea","seamos","sean","seas","segun","segunda","segundo","según","seis","ser","sera","seremos","será","serán","serás","seré","seréis","sería","seríais","seríamos","serían","serías","seáis","señaló","si","sido","siempre","siendo","siete","sigue","siguiente","sin","sino","sobre","sois","sola","solamente","solas","solo","solos","somos","son","soy","soyos","su","supuesto","sus","suya","suyas","suyo","suyos","sé","sí","sólo","t","tal","tambien","también","tampoco","tan","tanto","tarde","te","temprano","tendremos","tendrá","tendrán","tendrás","tendré","tendréis","tendría","tendríais","tendríamos","tendrían","tendrías","tened","teneis","tenemos","tener","tenga","tengamos","tengan","tengas","tengo","tengáis","tenida","tenidas","tenido","tenidos","teniendo","tenéis","tenía","teníais","teníamos","tenían","tenías","tercera","ti","tiempo","tiene","tienen","tienes","toda","todas","todavia","todavía","todo","todos","total","trabaja","trabajais","trabajamos","trabajan","trabajar","trabajas","trabajo","tras","trata","través","tres","tu","tus","tuve","tuviera","tuvierais","tuvieran","tuvieras","tuvieron","tuviese","tuvieseis","tuviesen","tuvieses","tuvimos","tuviste","tuvisteis","tuviéramos","tuviésemos","tuvo","tuya","tuyas","tuyo","tuyos","tú","u","ultimo","un","una","unas","uno","unos","usa","usais","usamos","usan","usar","usas","uso","usted","ustedes","v","va","vais","valor","vamos","van","varias","varios","vaya","veces","ver","verdad","verdadera","verdadero","vez","vosotras","vosotros","voy","vuestra","vuestras","vuestro","vuestros","w","x","y","ya","yo","z","él","éramos","ésa","ésas","ése","ésos","ésta","éstas","éste","éstos","última","últimas","último","últimos"]
#Texto se pone ne minúsculas
def text_lowercase(text):
	return text.lower()

# Eliminar los numeros o convertirlos a tedxto --> ingles
def remove_numbers(text):
	result = ''.join([i for i in text if not i.isdigit()])
	#result = re.sub(r' \d+ ', '', text)
	#result = re.sub(r' \d+', '', text)
	result = result.replace("  ", " ")
	return result

#Eliminar simbolos de puntuacion
def remove_punctuation(text):
	punctuation = f"{string.punctuation}{special_char}"
	translator = str.maketrans('', '', punctuation)
	text.translate(translator)
	#translator = str.maketrans('', '', special_char)
	return text.translate(translator)

# remove whitespace from text
def remove_whitespace(text, keepSpaces):
	output   = re.sub(r"[\n]*", "", text)
	if keepSpaces:
		return output
	return  " ".join(text.split())

#Stop words --> español ver listas y comparar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
# remove stopwords function
def remove_stopwords(text):
	stop_words = set(stopwords.words("spanish"))
	stop_words.update(set(stopWords))
	#stop_words = set(stopWords)
	word_tokens = word_tokenize(text)
	filtered_text = [word for word in word_tokens if word not in stop_words]

	return filtered_text

def read_files():
	for file in os.listdir():
		# Check whether file is in text format or not
		if file.endswith(".txt"):
			file_path = f"{path}/{file}"
			# call read text file function
			read_text_file(file_path)

def read_text_file(file_path):
	#os.chdir(path)
	with open(file_path, 'r',  encoding="utf8") as f:
		str_file = f.read()
		f.close()
		return str_file
 
def merge_corpus(politics_file,health_file, sport_file):
	for root, dirs, files in os.walk("./corpus/train", topdown=False):
		for name in files:
			#print(os.path.join(root, name))
			file_str = read_text_file(os.path.join(root, name))
			file_str = preprocess_corpus(file_str)
			if 'deportes' in root:
				sport_file.write(file_str)
			elif 'salud' in root:
				health_file.write(file_str)
			if 'politica' in root:
				politics_file.write(file_str)

def preprocess_corpus(file_str):
	file_str = text_lowercase(file_str)
	file_str = remove_numbers(file_str) 
	file_str = remove_punctuation(file_str)
	return file_str

def process_corpus(text_str, get_list, keepSpaces):
	text_str = remove_whitespace(text_str, keepSpaces)
	text_list_str = remove_stopwords(text_str)

	if get_list == False:
		text_str = list_to_string(text_list_str)
		return text_str
	else:
		return text_list_str

def list_to_string(lista):
	lista = ' '.join(str(e) for e in lista)
	return lista


#nltk.data.path.append("C:\\Users\\julia\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\nltk\\") 
#nltk.download('punkt')
