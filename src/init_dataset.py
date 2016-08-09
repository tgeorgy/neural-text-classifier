# Initial wikipedia pages processing

import re
from itertools import izip


def parse(t):
    # Take text before first section
    t2 = t.split('\n\n=')[0]
    t2 = t2.split('==')[0]

    # remove 'text' openning tag
    t2 = re.sub('<text.*?>', '', t2)

    # replace keywords
    t2 = re.sub('&lt;', '<', t2)
    t2 = re.sub('&gt;', '>', t2)
    t2 = re.sub('&quot;', '\"', t2)
    t2 = re.sub('&?nbsp;', '\"', t2)

    # remove tags content
    t2 = re.sub('<ref.*?\/>', '', t2)
    t2 = re.sub('<ref.*?>(\w|\W)*?</ref>', '', t2)
    t2 = re.sub('<sub.*?>(\w|\W)*?</sub>', '', t2)
    t2 = re.sub('<sup.*?>(\w|\W)*?</sup>', '', t2)
    t2 = re.sub('<math.*?>(\w|\W)*?</math>', '', t2)
    t2 = re.sub('<syntaxhighlight.*?>(\w|\W)*?</syntaxhighlight>', '', t2)
    t2 = re.sub('<small>', '', t2)
    t2 = re.sub('</?small>', '', t2)
    t2 = re.sub('<code>', '', t2)
    t2 = re.sub('</?code>', '', t2)

    # html comments
    t2 = re.sub('<!--(\w|\W)*?-->', '', t2)
    # remover first part of wiki links
    t2 = re.sub('(?<=\[\[)[^\]^:]*?\|(?=[^\]])', '', t2)
    # remove links in single square brackets
    t2 = re.sub('(?<=[^\[])(\[)[^\[]*]', '', t2)
    # remover wiki keywords
    t2 = re.sub('(\[\[)[^\]]*?:[^\]]*?(\]\])', '', t2)
    # remove double square brackets
    t2 = re.sub('(\[\[|\]\])', '', t2)

    t3 = ''
    acc = 0
    for c in t2:
        if c == '{':
            acc += 1
        elif c == '}':
            acc -= 1
        elif acc == 0:
            t3 += c

    t3 = re.sub('(\n|\t|\"|\'{2,3})', ' ', t3)
    t3 = re.sub('\s{2,10}', ' ', t3)

    return t3


# Parse text, save general description only
ifile = open('../data/enwiki-20160720-pages-articles.xml')
ofile_txt = open('../cache/text.txt', 'w')
ofile_id = open('../cache/id.txt', 'w')

i = 0
read_page = False
read_text = False

i_doc = 0
i_out = 0

for l in ifile:
    if '</page>' in l:
        read_page = False
        i_doc += 1

    if read_page:
        if '<id>' in l and len(id_) == 0:
            id_ = l.split('<id>')[1].split('</id>')[0]
        if '<ns>' in l:
            # checking page type
            isok = '<ns>0</ns>' in l
        if '<text' in l:
            read_text = True
            t = ''

        if '</text>' in l:
            read_text = False

            if len(t) > 0 and 'REDIRECT' not in t:
                txt = parse(t)

                if isok > 0:
                    ofile_id.write(id_)
                    ofile_id.write('\n')
                    ofile_txt.write(txt)
                    ofile_txt.write('\n')

                    i_out += 1

        if read_text:
            t += l

    if '<page>' in l:
        read_page = True
        id_ = ''

    i += 1
    if i % 4000000 == 0:
        print 'Documents parsed: ', i_doc
        print 'Documents written: ', i_out

ifile.close()
ofile_txt.close()
ofile_id.close()


# Parse page categories and save to file
start_str = 'INSERT INTO `categorylinks` VALUES'
cat = []
with open('../data/enwiki-20160720-categorylinks.sql') as f:
    for l in f:
        if l[:len(start_str)] == start_str:
            l2 = l[len(start_str):-2]
            l2 = '['+l2+']'
            tmp = eval(l2)
            tmp = [(row[0], row[1]) for row in tmp if row[6] == 'page']
            cat += tmp

with open('../cache/cat.txt', 'w') as ofile_cat:
    for c in cat:
        ofile_cat.write(str(c[0])+','+c[1]+'\n')


# Filter page - category pairs by available page ids
val_page_ids = []
with open('../cache/id.txt') as f:
    for l in f:
        val_page_ids.append(l[:-1])
val_page_ids = set(val_page_ids)

page_cat = {}
with open('../cache/cat.txt') as f:
    for l in f:
        l2 = l.split(',')
        page_i = l2[0]
        if page_i in val_page_ids:
            cat_i = l[len(page_i)+1:-1]
            if page_i in page_cat:
                page_cat[page_i].append(cat_i)
            else:
                page_cat[page_i] = [cat_i]

page_cat_tmp = {}
for k, v in page_cat.iteritems():
    page_cat_tmp[k] = sorted(v)


# Filter categories by frequency
cat_acc = {}
for k, v in page_cat_tmp.iteritems():
    for vi in v:
        if vi in cat_acc:
            cat_acc[vi] += 1
        else:
            cat_acc[vi] = 1

cats_to_remove = [k for k, v in cat_acc.iteritems() if v < 100]
cats_to_remove_s = set(cats_to_remove)


# Filter categories by manually collected keywords
kw = ['-century', 'template', 'player', '_using_', 'alumni', 'Alumni',
      'inventors', 'Members', '_martyrs', 'Articles', 'people',
      'People', 'actresses', 'redirects', 'Articles', 'pages',
      'personnel', 'journalists', 'personalities', 'broadcasters',
      'officer', 'Actresses', 'name', 'Redirects', '_pages_', 'Award',
      'stubs', '_stub', 'parameter', 'Records_albums', 'missing',
      'articles', 'episodes', 'Debut', 'albums', '[0-9]{3,4}',
      'Wikidata', 'source', 'CS1', 'Pages', 'award', 'Hadeninae',
      'deaths', 'medalists', 'usage', 'lacking', 'unknown', '-language',
      'Australian_rules_footballers', 'English_footballers',
      'IUCN_Red_List_least_concern_species', 'coordinate', 'infobox',
      'Major_League_Baseball_pitchers', 'Russian_footballers',
      'Set_indices_on_ships', 'disputes', 'Association_football',
      'lists', 'English-language_films', 'Indian_films', 'Recipients',
      'Infobox', 'American_films', 'districts', 'Wikipedia',
      'uncertain', 'Spanish_footballers', 'Italian_footballers',
      'Manga_series', 'Windows_games', '_needing_confirmation',
      'IUCN_Category_II', '_footballers', 'isambiguation',
      'Unincorporated', 'Football_kits_with_incorrect_pattern',
      'Chemboxes', 'IUCN', 'eference', 'Communes', 'Set_indices',
      'Lamiinae', 'Spilomelinae', 'Pyramidellidae', 'Bulbophyllum',
      'Megachile', 'Megachile', 'Lists_', 'Eupithecia', 'ownships',
      'English-language', 'Muricidae', 'Coleophora', 'Pyraustinae',
      'Lithosiini', 'Towns', 'Cities', 'Calpinae', 'Mordellistena',
      'Archipini', 'Arctiinae', 'Geometridae', 'Phycitini', 'Scopula',
      'Villages', 'Major_League_Baseball_outfielders', 'Years',
      'Municipalities', 'Botanists_with_author_abbreviations',
      'Paramount_Pictures_films', 'Phaegopterina', 'Acanthocinini',
      'Drepanidae', 'Notodontidae', 'Conus', 'Albums', 'EC', 'century']
kw = '('+'|'.join(kw)+')'

for k in cat_acc.keys():
    if k in cats_to_remove_s:
        continue
    if re.findall(kw, k):
        cats_to_remove.append(k)

cats_to_remove_s = set(cats_to_remove)


# Select smallest categories for each page
# (making assumption that they are describing topic
# more precisely), remove rarely selected categories
for iter_i in range(4):
    ccat_acc = {}
    for k, v in page_cat_tmp.iteritems():
        tmp = 1000000
        v2 = ''
        for vi in v:
            if (vi not in cats_to_remove_s and
                    cat_acc[vi] < tmp):
                v2 = vi
                tmp = cat_acc[vi]
        if v2 in ccat_acc:
            ccat_acc[v2] += 1
        else:
            ccat_acc[v2] = 1
    cats_to_remove += [k for k, v in ccat_acc.iteritems() if v < 100]
    cats_to_remove_s = set(cats_to_remove)

    print 'Number of selected categories ',
    print len(cat_acc) - len(cats_to_remove_s)


# Combine results
page_cat = {}
for k, v in page_cat_tmp.iteritems():
    c = ''
    tmp = 1000000
    for vi in v:
        if vi in cats_to_remove_s:
            continue
        if cat_acc[vi] < tmp:
            c = vi
            tmp = cat_acc[vi]
    if c != '':
        page_cat[k] = c


# Export results
fi_txt = open('../cache/text.txt', 'r')
fi_pid = open('../cache/id.txt', 'r')
fo_txt = open('../cache/text_selected.txt', 'w')
fo_cat = open('../cache/cat_selected.txt', 'w')


for txt, pid in izip(fi_txt, fi_pid):
    pid = pid[:-1]
    if pid in page_cat:
        fo_txt.write(txt)
        fo_cat.write(page_cat[pid]+'\n')


fi_txt.close()
fi_pid.close()
fo_txt.close()
fo_cat.close()
