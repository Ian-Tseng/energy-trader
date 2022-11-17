# Energy trader

### Idea
1. 
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


### Model architecture


### Action driver

```
1. Get request supply- rate
    1) Idea: The price depends on request_of_customers/ productivity_of_product. 
    Reference url: https://www.youtube.com/watch?v=PHe0bXAIuk0
    2) We try to find the best price rate based on above request-supply rate.
    3) The following is the table show average request-supply rate () in 
    request-supply rate= $$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$



2. 
```
