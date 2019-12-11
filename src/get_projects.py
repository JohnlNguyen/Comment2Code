import requests

base_url = 'http://api.github.com/search/repositories?q=language:java&sort=stars&per_page=100&page='
unique_projects = {}
for page in range(1, 11):
    url = base_url + str(page)
    r = requests.get(url)
    data = r.json()
    data = data.get("items", [])
    for d in data:
        # only using active repos
        if d['archived']:
            continue
        unique_projects[d["full_name"]] = int(d["stargazers_count"])

with open("../data/projects.txt", "w") as f:
    projects = sorted(unique_projects.items(),
                      key=lambda e: e[1], reverse=True)
    for name, stars in projects:
        f.write("{0}\t{1}\n".format(name, stars))
