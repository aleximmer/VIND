library(feather)
library(locits)

df <- read_feather('data/stock_weekly.feather')
mdf <- data.frame(df)

for (stock in 1:29){
	ans <- hwtos2(mdf[103:358, stock])
	print(ans)
}
