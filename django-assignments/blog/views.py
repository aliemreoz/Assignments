from django.contrib.auth.decorators import permission_required
from django.core.exceptions import PermissionDenied
from django.shortcuts import render

# Create your views here.
from django.http import Http404
from django.http import HttpResponse
from django.shortcuts import render

from tags.models import Tag
from .models import Blog
from .forms import BlogForm

def show_entries(request):

    if request.method == "POST":
        form = BlogForm(request.POST)
        if form.is_valid():
            blog = form.save(commit=False)
            blog.owner = request.user
            blog.save()
            form.save_m2m()

    elif request.method == "GET":
        form = BlogForm()

    return render(request, "my_entries.html", {"entries": Blog.objects.filter(owner=request.user.id),
                                             "tags":Tag.objects.all(),
                                             "form": form})
	
def get_entries(request, entry_id):
    try:
        blog = Blog.objects.get(id=int(entry_id))
        if request.user.id != blog.owner.id:
            raise PermissionDenied
        return render(request, "detailed_entries.html", {"blog": blog})
    except Blog.DoesNotExist:
        raise Http404("There is no director.")
		
@permission_required('is_superuser')
def show_all_entries(request):
    return render(request, "my_entries.html", {"entries": Blog.objects.all()})

@permission_required('is_superuser')
def show_all_entries_from_user(request, userId):
    return render(request, "my_entries.html", {"entries": Blog.objects.filter(owner=userId)})
