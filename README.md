# Energy trader

### Idea
1. The solar device generate less energy or higher energy in specific time and a consumption pattern that show a wave in specific time. 
2. According above patterns , find the best price rate in specific time. The formula: price rate=  request volume/ generate volume.
3. Logic of agent : Start sell action if price rate lower than a threshold. Start buy action if price rate higher than a threshold.


### Environment
```
The following is the steps to build pipenv environment:
$ cd energy_trader
$ pipenv install 
$ pipenv shell
$ python --3.8

```
### Usage

```
$ python main.py --generation --consumption --output
    
```


