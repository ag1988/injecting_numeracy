# Numeric data (ND) generation 

**gen_numeric_data.py is used to create numeric synthetic data (ND) with the following skills:**
(numbers can be floats with at most 2 decimal points or sometimes only integers)
1. 19517.4 - 17484 - 10071.75 + 1013.21 : -7025.14   (#args can be 2,3,4)
2. most/least/superlative(1072.1, 17938, 5708.65, 14739.16) : 17938
3. argmin/argmax(toppy 8105.5, cockney 7111.0, nickelic 1463.16, tiredom 6929) : toppy
4. most recent/least recent/superlative(July 16, 134; June 23, 134; 24 July 134; 28 October 134) : 28 October 134
5. difference in days/months/years(April 21, 1381; 13 April 1381) : 7
6. percent not photochemist, floodgate, retiringly :: photochemist 0.82%, morningward 54.4%, floodgate 2.0%, reline 0.78%, retiringly 42% : 55.18
  
```
python gen_numeric_data.py --num_samples 1e6 --num_dev_samples 1e4 --output_jsonl ../data/synthetic_numeric.jsonl
```

This will save `synthetic_numeric.jsonl` in `../data` dir.  


Contact: Ankit Gupta