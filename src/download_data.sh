mkdir ../data ../cache
cd ../data
wget https://dumps.wikimedia.org/enwiki/20160720/enwiki-20160720-categorylinks.sql.gz
wget https://dumps.wikimedia.org/enwiki/20160720/enwiki-20160720-pages-articles.xml.bz2

gzip -d enwiki-20160720-categorylinks.sql.gz
bzip2 -d enwiki-20160720-pages-articles.xml.bz2
