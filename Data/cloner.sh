cat projects.txt | xargs -P8 -n1 -I% bash -c 'echo %; \
 name=$(echo % | cut -d" " -f1); \
 head=$(echo $name | cut -d"/" -f1); \
 DIR=Repos/$head; mkdir -p $DIR; \
 git clone -q https://github.com/$name Repos/$name;'