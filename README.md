# hungyu
#python_self_learning

這是一個自我練習，練習如何使用python進行網路爬蟲，進而找到所需的資料

from selenium import webdriver

from selenium.webdriver.common.by import By

from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC

#啟動瀏覽器

driver = webdriver.Chrome()

#載入網頁

url = "https://eap.lib.ncku.edu.tw/NewArrivals/NewArrivals.php"

driver.get(url)

try:

    #使用顯式等待等待特定元素加載
    
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "css_td_2")))
    
    #提取書名
    
    book_elements = driver.find_elements(By.CLASS_NAME, "css_td_2")
    
    for element in book_elements:
        title = element.find_element(By.TAG_NAME, "a").text
        print("書名:", title)
except Exception as e:
    print("發生錯誤:", str(e))
finally:
    # 關閉瀏覽器
    driver.quit()
