from tests.convert.regression_base import RegressTest

if __name__ == '__main__':
    #compare_cases = ['test05', 'test09', 'test10', 'test10b', 'Calleg', 'GRW_Plaster', 'ZRW_WestIndian']
    #compare_cases = ['test10']  # control test cases for comparison
    compare_cases = ['GLWACSO']  

    my_test = RegressTest(compare_cases)
    my_test.run_test()