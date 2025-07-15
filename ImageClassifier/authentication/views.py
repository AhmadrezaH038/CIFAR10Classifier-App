from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from .forms import UserRegistrationForm, UserLoginForm



def Login(request):
    if request.user.is_authenticated:
        return redirect("frontend:home")
    if request.method == "POST":
        form = UserLoginForm(request=request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"welcome.")
            return redirect('frontend:home')
        else:
            messages.error(request, "Invalid credentials.")
    else:
        form = UserLoginForm()
    return render(request, 'authentication/login.html', {'form': form})


def user_logout(request):
    logout(request)
    messages.info(request, "You've been logged out.")
    return redirect("frontend:home")



def SignUp(request):
    if request.user.is_authenticated:
        return redirect('frontend:home')
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration Successful")
            return redirect("frontend:home")
    else :
        form = UserRegistrationForm()
    
    return render(request, 'authentication/signup.html', {"form": form})

