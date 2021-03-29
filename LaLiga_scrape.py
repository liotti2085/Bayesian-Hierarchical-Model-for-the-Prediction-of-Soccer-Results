from urllib.request import urlopen as uReq
from urllib.error import HTTPError
from bs4 import BeautifulSoup as soup
import numpy as np
import pandas as pd

weeks = range(1, 39) #35 upper limit when ready
print('Enter End of Season Year:')
year_2 = int(input())
year_1 = year_2 - 1
string = 'https://www.worldfootball.net/schedule/esp-primera-division-{}-{}-spieltag/{}'

home_teams = []
home_scores =[]
away_teams = []
away_scores =[]
home_index = []
away_index = []
matches = []
match = 1

for week in weeks:
	url = string.format(year_1, year_2, week)

	# open connection, grab page
	try:		
		uClient = uReq(url)
		page_html = uClient.read()
		uClient.close()
			
	except HTTPError as e:
		continue

	else:
		page_soup = soup(page_html, "html.parser")
		table = page_soup.select('table')[1]

		table_rows = table.find_all('tr')

		for tr in table_rows:
			home_teams.append(tr.findChildren('a')[0].text)
			away_teams.append(tr.findChildren('a')[1].text)

			#home_index.append(mydict[home_teams[-1]])
			#away_index.append(mydict[away_teams[-1]])

			score = tr.findChildren('a')[2].text
			home_scores.append(score[0])
			away_scores.append(score[2])

			matches.append(match)
			match = match + 1

teams = sorted(np.unique(home_teams))
mydict = {}
for i in range(0, len(teams)):
	mydict[teams[i]] = i + 1

for i in range(0, len(home_teams)):
	home_index.append(mydict[home_teams[i]])
	away_index.append(mydict[away_teams[i]])

scores = pd.DataFrame({
	'g' : matches,
	'home_team' : home_teams,
	'away_team' : away_teams,
	'h(g)' : home_index,
	'a(g)' : away_index,
	'home_goals' : home_scores,
	'away_goals' : away_scores
})

scores.to_csv('scores.csv', index = False)