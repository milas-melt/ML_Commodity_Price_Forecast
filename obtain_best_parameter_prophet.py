import tool

if __name__ == '__main__':
    train, test = tool.make_monthly_data('IndexPrices__US792.xlsx')
    tool.get_best_para(train)
