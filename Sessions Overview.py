# Sessions Overview

# How many unique actions?
sessions['action'].nunique()  # 331
sessions['action_type'].nunique()  # 9 
sessions['action_detail'].nunique()  # 128

# Top?
sessions.groupby(['action'])['action'].count().sort_values(ascending=False).head(15)
'''
show                     1317604
index                     468168
personalize               454755
search_results            425405
ajax_refresh_subtotal     318871
similar_listings          311480
update                    210240
search                    208734
'''

sessions.groupby(['action_type'])['action_type'].count().sort_values(ascending=False).head(15)
'''
view                1688709
data                1202837
click               1076270
-unknown-            569353
submit               362927
message_post          55933
booking_request       10694
partner_callback       7054
booking_response          2
'''

sessions.groupby(['action_detail'])['action_detail'].count().sort_values(ascending=False).head(15)
'''
view_search_results            923617
p3                             617708
-unknown-                      569353
wishlist_content_update        454755
change_trip_characteristics    318871
similar_listings               311480
user_profile                   271801
update_listing                 158553
user_social_connections        144955
header_userpic                  86047
listing_reviews                 84579
message_thread                  78306
user_wishlists                  77580
dashboard                       74649
contact_host                    57446
'''

# Combined
sessions.groupby(['action', 'action_detail'])['action', 'action_detail'].count().sort_values(
    'action', ascending=False).head(15)
'''
show                  p3                           616946         616946
personalize           wishlist_content_update      454755         454755
search_results        view_search_results          425405         425405
ajax_refresh_subtotal change_trip_characteristics  318871         318871
similar_listings      similar_listings             311480         311480
index                 view_search_results          292501         292501
show                  user_profile                 271801         271801
search                view_search_results          205342         205342
update                update_listing               158553         158553
social_connections    user_social_connections      144955         144955
active                -unknown-                     86138          86138
header_userpic        header_userpic                86047          86047
reviews               listing_reviews               84579          84579
dashboard             dashboard                     74649          74649
index                 -unknown-                     73718          73718
'''

sessions.groupby(['action', 'action_detail', 'action_type'])['action', 
    'action_detail', 'action_type'].count().sort_values(
    'action', ascending=False).head(15)