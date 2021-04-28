from tests.convert.regression_base import RegressTestBase


class ConversionRegressionTest(RegressTestBase):
    def __init__(self):
        super(ConversionRegressionTest, self).__init__()

if __name__ == '__main__':
    my_test = ConversionRegressionTest()
    my_test.run_test()