from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# set chromedriver.exe path
PATH = "/Users/pushkarambastha/Encrypted Files/E drive/DS,AI,ML/Selenium_tutorial/chromedriver"
driver = webdriver.Chrome(PATH)
driver.implicitly_wait(0.5)
# maximize browser
driver.maximize_window()
# launch URL
driver.get("https://codepen.io/sumanth2303/pen/PoGjWwd?editors=0010")
# # open file in write and binary mode
# with open('Logo.png', 'wb') as file:
#     # identify image to be captured
#     l = driver.find_element_by_xpath('//*[@alt="Tutorialspoint"]')
#     # write file
#     file.write(l.screenshot_as_png)
# # close browser
# driver.quit()

try:

    # open file in write and binary mode
    with open('Logo.png', 'wb') as file:
        image = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "generated-captcha"))
        )
        img = driver.find_element(By.XPATH, '//*[@id ="generated-captcha"]')
        # write file
        file.write(img.screenshot_as_png)
        driver.save_screenshot("screenshot.png")
        # images = driver.find_elements(By.ID, "generated-captcha")
        # for image in images:
        #     print(image.get_attribute('src'))

finally:
    driver.quit()
