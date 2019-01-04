/* 
 * Widgets for Social Network photo stream.
 * 
 * Author: Pixel Industry
 * Website: http://pixel-industry.com
 * Version: 1.4
 *
 */

(function ($) {
    $.fn.socialstream = function (options) {
        var defaults = {
            socialnetwork: 'flickr',
            username: 'pixel-industry',
            limit: 6,
            overlay: true,
            apikey: false,
            accessToken: '',
            picasaAlbumId: ''
        };
        var options = $.extend(defaults, options);
        return this.each(function () {
            var object = $(this);
            switch (options.socialnetwork) {
                case 'flickr':
                    object.append("<ul class=\"flickr-list\"></ul>")
                    $.getJSON("https://api.flickr.com/services/rest/?method=flickr.people.findByUsername&username=" + options.username + "&format=json&api_key=32ff8e5ef78ef2f44e6a1be3dbcf0617&jsoncallback=?", function (data) {
                        var user_id = data.user.nsid;
                        $.getJSON("https://api.flickr.com/services/rest/?method=flickr.photos.search&user_id=" + user_id + "&format=json&api_key=85145f20ba1864d8ff559a3971a0a033&per_page=" + options.limit + "&page=1&extras=url_sq&jsoncallback=?", function (data) {
                            $.each(data.photos.photo, function (num, photo) {
                                var photo_author = photo.owner;
                                var photo_title = photo.title;
                                var photo_src = photo.url_sq;
                                var photo_id = photo.id;
                                var photo_url = "https://www.flickr.com/photos/" + photo_author + "/" + photo_id;
                                var photo_container = $('<img/>').attr({
                                    src: photo_src,
                                    alt: photo_title
                                });
                                var url_container = $('<a/>').attr({
                                    href: photo_url,
                                    target: '_blank',
                                    title: photo_title
                                });

                                var tmp = $(url_container).append(photo_container);
                                if (options.overlay) {
                                    var overlay_div = $('<div/>').addClass('img-overlay');
                                    $(url_container).append(overlay_div);
                                }
                                var li = $('<li/>').append(tmp);
                                $("ul", object).append(li);
                            })
                        });
                    });
            }
        });
    };
})(jQuery);