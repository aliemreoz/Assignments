from django.conf.urls import url

from .views import show_entries, get_entries, show_all_entries, show_all_entries_from_user

urlpatterns = [
    url(r'^$', show_entries),
    url(r'^(?P<entry_id>[0-9]+)', get_entries),
	url(r'^all/$', show_all_entries),
    url(r'^all/user/(?P<userId>[0-9]+)$', show_all_entries_from_user)
]
