import wikipediaapi, re, random, sys

wiki = wikipediaapi.Wikipedia('Wikipedia Scraper (dingberjer@gmail.com)', 'en')
def main():
    x = input("Would you like to know about a |specific| event, a |random| event, or an event in a |general| year? ")
    if x.lower() == "specific":
        print(specific_event(input("What would you like to know about? ")))
    elif x.lower() == "random":
        randomlist = []
        battles = wiki.page("List of World War II Battles").links
        for link in battles.keys():
            randomlist.append(link)
        i = random.randint(0, len(randomlist) - 1)
        print(random_event(i))
    elif x.lower() == "general":
        print(general_event())
    else:
        print("Please enter |specific|, or |random|, or |general| to continue")

def specific_event(i):
    page = wiki.page(i)
    if page.exists():
        if matches := re.search(r"1939|194[0-5]", page.summary):
            return page.summary
        else:
            return "Page is not contained in our catalogues"
    else:
        return "Page does not exist"

def random_event(x):
    randomlist = []
    battles = wiki.page("List of World War II battles").links
    for link in battles.keys():
        randomlist.append(link)
    battle_page = wiki.page(randomlist[x])
    return battle_page.summary

def general_event():
    generallist = []
    portal = wiki.page("Category: World War II").categorymembers
    for category in portal.keys():
        generallist.append(category.replace("Category:",""))
    print(generallist)
    while True:
        inputted = input("")
        if inputted in generallist:
            generallist = []
            portal = wiki.page(f"Category:{inputted}").categorymembers
            for category in portal.keys():
                generallist.append(category.replace("Category:",""))
            if generallist == []:
                return wiki.page(inputted).summary
            else:
                print(generallist)
        else:
            sys.exit("Incorrect selection")


if __name__ == "__main__":
    main()
