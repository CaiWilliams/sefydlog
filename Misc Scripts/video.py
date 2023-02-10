import cv2
import os
import datetime
import pandas as pd

image_folder = 'images'
video_name = 'video.avi'

start_date = datetime.datetime(2010,1,1,0)
end_date = datetime.datetime(2010,12,31,21)
dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()

def fetch_images(dates,gas):
    images = []
    for date in dates:
        folder = os.path.join(os.getcwd(),str(date.year),str(date.month),str(date.day),str(date.hour))
        for img in os.listdir(folder):
            if gas in img:
                images.append(os.path.join(os.getcwd(),str(date.year),str(date.month),str(date.day),str(date.hour),img))
    return images

def write_images(name, images):
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(name, 0, 16, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


images = fetch_images(dates,'air_temperature')
write_images('2010_air_temperature.avi',images)

images = fetch_images(dates,'carbon_dioxide')
write_images('2010_carbon_dioxide.avi',images)

images = fetch_images(dates,'carbon_monoxide')
write_images('2010_carbon_monoxide.avi',images)

images = fetch_images(dates,'formaldehyde')
write_images('2010_formaldehyde.avi',images)

images = fetch_images(dates,'methane')
write_images('2010_methane.avi',images)

images = fetch_images(dates,'nitric_acid')
write_images('2010_nitric_acid.avi',images)

images = fetch_images(dates,'nitrogen_dioxide')
write_images('2010_nitrogen_dioxide.avi',images)

images = fetch_images(dates,'ozone')
write_images('2010_ozone.avi',images)

images = fetch_images(dates,'relative_humidity')
write_images('2010_relative_humidity.avi',images)

images = fetch_images(dates,'sulfur_dioxide')
write_images('2010_sulfur_dioxide.avi',images)

images = fetch_images(dates,'surface_air_pressure')
write_images('2010_surface_air_pressure.avi',images)

images = fetch_images(dates,'TAU550')
write_images('2010_TAU550.avi',images)

images = fetch_images(dates,'water_vapour')
write_images('2010_water_vapour.avi',images)