from datetime import datetime

from tests.convert.regression_base import RegressTest 
if __name__ == '__main__':
    #compare_cases = ['test05', 'test09', 'test10', 'test10b', 'Calleg', 'GRW_Plaster', 'ZRW_WestIndian']
    #compare_cases = ['test10']  # control test cases for comparison
    compare_case = 'GLWACSO'  

    start = datetime.now()

    # test = RegressTest(compare_case, ids=['434'],activites=['GQUAL'],threads=1)  
    test = RegressTest('test10')
    results = test.run_test()
    test.generate_report(test.html_file, results)

    print(f'runtime: {datetime.now() - start}')