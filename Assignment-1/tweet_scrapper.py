import twint

#configuration
config = twint.Config()
config.Search = ["pollution in India","India"]
config.Limit = 30000
config.Store_csv = True
config.Output = "Pollution_in_india_Ayush_latest-2.csv"
# config.Location=True
# config.Near = "India"
# config.Since = "2014-12-27"
config.Until = "2021-04-29"
config.Hide_output = True
#config.geo="India"

#running search
twint.run.Search(config)