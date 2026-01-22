from pydantic import BaseModel

class student(BaseModel):
    name : str 

new_student = {
    'name' : 'John Doe'
}

student_obj = student(**new_student)
print(student_obj)