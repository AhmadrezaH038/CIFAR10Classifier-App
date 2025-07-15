from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
from .inferences import classify_image

@login_required
@require_http_methods(['GET', 'POST'])
def run_classifier(request):
    result = None
    if request.method == "POST" and request.FILES.get('image'):
        result = classify_image(request.user, request.FILES['image'])
    return render(request, 'classifier.html', {"result": result})
