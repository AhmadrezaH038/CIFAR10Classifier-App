from django.shortcuts import render
from classifier.models import RecentAction
from django.contrib.auth.decorators import login_required


@login_required
def classifier_page(request):
    return render(request, 'base.html')


@login_required
def dashboard(request):
    actions = (
        RecentAction.objects.filter(user=request.user).order_by('-timestamp')[:10]
    )
    return render(request, 'dashboard.html', {'actions': actions})
